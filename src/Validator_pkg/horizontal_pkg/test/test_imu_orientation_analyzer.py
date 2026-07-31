#!/usr/bin/env python3

import csv
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stderr, redirect_stdout
import io

from horizontal_pkg import imu_orientation_analyzer as analyzer


class QuaternionConversionTest(unittest.TestCase):
    def convert(self, *values):
        conversion = getattr(analyzer, "quaternion_to_roll_pitch", None)
        self.assertIsNotNone(
            conversion,
            "quaternion_to_roll_pitch is not implemented",
        )
        return conversion(*values)

    def test_identity_quaternion_is_level(self):
        result = self.convert(0.0, 0.0, 0.0, 1.0)

        self.assertEqual(result.quaternion_norm, 1.0)
        self.assertEqual(result.roll_rad, 0.0)
        self.assertEqual(result.pitch_rad, 0.0)
        self.assertFalse(result.normalization_warning)

    def test_positive_roll_is_recovered(self):
        half_angle = math.radians(10.0) / 2.0

        result = self.convert(
            math.sin(half_angle),
            0.0,
            0.0,
            math.cos(half_angle),
        )

        self.assertAlmostEqual(math.degrees(result.roll_rad), 10.0, places=10)
        self.assertAlmostEqual(result.pitch_rad, 0.0, places=10)

    def test_negative_pitch_is_recovered(self):
        half_angle = math.radians(-10.0) / 2.0

        result = self.convert(
            0.0,
            math.sin(half_angle),
            0.0,
            math.cos(half_angle),
        )

        self.assertAlmostEqual(result.roll_rad, 0.0, places=10)
        self.assertAlmostEqual(math.degrees(result.pitch_rad), -10.0, places=10)

    def test_quaternion_sign_does_not_change_orientation(self):
        positive = self.convert(0.1, -0.2, 0.3, 0.9)
        negative = self.convert(-0.1, 0.2, -0.3, -0.9)

        self.assertAlmostEqual(negative.roll_rad, positive.roll_rad, places=12)
        self.assertAlmostEqual(negative.pitch_rad, positive.pitch_rad, places=12)

    def test_non_unit_quaternion_is_normalized_and_warned(self):
        result = self.convert(0.0, 0.0, 0.0, 2.0)

        self.assertEqual(result.quaternion_norm, 2.0)
        self.assertEqual(result.roll_rad, 0.0)
        self.assertEqual(result.pitch_rad, 0.0)
        self.assertTrue(result.normalization_warning)

    def test_pitch_at_inverse_sine_boundary_is_finite(self):
        half_angle = math.pi / 4.0

        result = self.convert(
            0.0,
            math.sin(half_angle),
            0.0,
            math.cos(half_angle),
        )

        self.assertTrue(math.isfinite(result.pitch_rad))
        self.assertAlmostEqual(result.pitch_rad, math.pi / 2.0, places=10)

    def test_zero_norm_quaternion_is_rejected(self):
        error_type = getattr(analyzer, "InvalidQuaternionError", ValueError)

        with self.assertRaises(error_type):
            self.convert(0.0, 0.0, 0.0, 0.0)

    def test_non_finite_quaternion_is_rejected(self):
        error_type = getattr(analyzer, "InvalidQuaternionError", ValueError)

        with self.assertRaises(error_type):
            self.convert(math.nan, 0.0, 0.0, 1.0)


class SampleConsolidationTest(unittest.TestCase):
    def raw_sample(
        self,
        header_stamp_ns,
        bag_record_stamp_ns,
        seq,
        frame_id="IMU-1",
        quaternion=(0.0, 0.0, 0.0, 1.0),
    ):
        sample_type = getattr(analyzer, "RawOrientationSample", None)
        self.assertIsNotNone(
            sample_type,
            "RawOrientationSample is not implemented",
        )
        return sample_type(
            header_stamp_ns=header_stamp_ns,
            bag_record_stamp_ns=bag_record_stamp_ns,
            seq=seq,
            frame_id=frame_id,
            qx=quaternion[0],
            qy=quaternion[1],
            qz=quaternion[2],
            qw=quaternion[3],
        )

    def example_result(self):
        consolidate = getattr(analyzer, "consolidate_samples", None)
        self.assertIsNotNone(
            consolidate,
            "consolidate_samples is not implemented",
        )
        samples = [
            self.raw_sample(20, 200, 1),
            self.raw_sample(10, 210, 2),
            self.raw_sample(20, 220, 3, frame_id="IMU-new"),
            self.raw_sample(0, 230, 4),
            self.raw_sample(30, 240, 5, quaternion=(0.0, 0.0, 0.0, 0.0)),
        ]
        return consolidate(samples)

    def test_samples_are_chronological_unique_and_last_duplicate_wins(self):
        result = self.example_result()

        self.assertEqual(
            [row.raw.header_stamp_ns for row in result.rows],
            [10, 20],
        )
        self.assertEqual(result.rows[1].raw.seq, 3)
        self.assertEqual(result.rows[1].raw.frame_id, "IMU-new")

    def test_sample_quality_counters_are_preserved(self):
        result = self.example_result()

        self.assertEqual(result.stats.total_messages, 5)
        self.assertEqual(result.stats.duplicate_replacements, 1)
        self.assertEqual(result.stats.timestamp_reversals, 1)
        self.assertEqual(result.stats.zero_stamp_count, 1)
        self.assertEqual(result.stats.invalid_quaternion_count, 1)
        self.assertEqual(result.stats.normalization_warning_count, 0)
        self.assertEqual(result.stats.frame_ids, ("IMU-1", "IMU-new"))

    def test_no_valid_samples_is_an_error(self):
        consolidate = getattr(analyzer, "consolidate_samples", None)
        self.assertIsNotNone(
            consolidate,
            "consolidate_samples is not implemented",
        )
        samples = [
            self.raw_sample(0, 100, 1),
            self.raw_sample(10, 110, 2, quaternion=(0.0, 0.0, 0.0, 0.0)),
        ]

        with self.assertRaisesRegex(
            analyzer.AnalyzerError,
            "no valid IMU orientation samples",
        ):
            consolidate(samples)


class CsvExportTest(SampleConsolidationTest):
    EXPECTED_FIELDS = [
        "header_stamp_ns",
        "elapsed_s",
        "bag_record_stamp_ns",
        "seq",
        "frame_id",
        "qx",
        "qy",
        "qz",
        "qw",
        "quaternion_norm",
        "roll_rad",
        "pitch_rad",
        "roll_deg",
        "pitch_deg",
    ]

    def test_csv_schema_and_elapsed_time_are_exact(self):
        write_csv = getattr(analyzer, "write_csv_atomic", None)
        self.assertIsNotNone(
            write_csv,
            "write_csv_atomic is not implemented",
        )
        result = self.example_result()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "attitude.csv"
            write_csv(result.rows, output)
            with output.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)

        self.assertEqual(reader.fieldnames, self.EXPECTED_FIELDS)
        self.assertEqual(
            [int(row["header_stamp_ns"]) for row in rows],
            [10, 20],
        )
        self.assertEqual(float(rows[0]["elapsed_s"]), 0.0)
        self.assertEqual(float(rows[1]["elapsed_s"]), 1e-8)

    def test_failed_replace_preserves_destination_and_removes_temporary_file(self):
        write_csv = getattr(analyzer, "write_csv_atomic", None)
        self.assertIsNotNone(
            write_csv,
            "write_csv_atomic is not implemented",
        )
        result = self.example_result()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attitude.csv"
            output.write_text("previous\n", encoding="utf-8")
            files_before = set(output.parent.iterdir())
            with mock.patch.object(
                os,
                "replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    write_csv(result.rows, output)

            self.assertEqual(output.read_text(encoding="utf-8"), "previous\n")
            self.assertEqual(set(output.parent.iterdir()), files_before)


class FakeTime:
    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds

    def to_nsec(self):
        return self.nanoseconds


class FakeHeader:
    def __init__(self, stamp_ns, seq, frame_id="IMU-1"):
        self.stamp = FakeTime(stamp_ns)
        self.seq = seq
        self.frame_id = frame_id


class FakeOrientation:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class FakeImu:
    _type = "sensor_msgs/Imu"

    def __init__(self, stamp_ns, seq, frame_id="IMU-1"):
        self.header = FakeHeader(stamp_ns, seq, frame_id)
        self.orientation = FakeOrientation()


class FakeWrongMessage(FakeImu):
    _type = "geometry_msgs/PoseStamped"


def bag_factory_for(records):
    class FakeBag:
        def __init__(self, _path, _mode):
            pass

        def __enter__(self):
            return self

        def __exit__(self, _error_type, _error, _traceback):
            return False

        def read_messages(self, topics):
            for topic, message, record_time in records:
                if topic in topics:
                    yield topic, message, record_time

    return FakeBag


class RosbagAnalysisTest(unittest.TestCase):
    def analyze(self, records, topic="/imu"):
        analyze_bag = getattr(analyzer, "analyze_bag", None)
        self.assertIsNotNone(analyze_bag, "analyze_bag is not implemented")
        return analyze_bag(
            Path("input.bag"),
            topic,
            bag_factory=bag_factory_for(records),
        )

    def test_two_imu_messages_become_two_rows(self):
        result = self.analyze(
            [
                ("/imu", FakeImu(10, 1), FakeTime(100)),
                ("/imu", FakeImu(20, 2), FakeTime(200)),
            ]
        )

        self.assertEqual(result.stats.total_messages, 2)
        self.assertEqual(len(result.rows), 2)
        self.assertEqual(result.rows[1].raw.bag_record_stamp_ns, 200)

    def test_missing_topic_names_topic_and_bag(self):
        with self.assertRaisesRegex(
            analyzer.AnalyzerError,
            r"topic /imu was not found in input\.bag",
        ):
            self.analyze([])

    def test_incompatible_message_names_required_type(self):
        with self.assertRaisesRegex(
            analyzer.AnalyzerError,
            "sensor_msgs/Imu",
        ):
            self.analyze(
                [("/imu", FakeWrongMessage(10, 1), FakeTime(100))]
            )

    def test_summary_contains_counts_frames_and_angle_ranges(self):
        result = self.analyze(
            [
                ("/imu", FakeImu(10, 1), FakeTime(100)),
                ("/imu", FakeImu(20, 2), FakeTime(200)),
            ]
        )
        format_summary = getattr(analyzer, "format_summary", None)
        self.assertIsNotNone(
            format_summary,
            "format_summary is not implemented",
        )

        summary = format_summary(
            result,
            Path("input.bag"),
            "/imu",
            Path("output.csv"),
        )

        self.assertIn("total_messages=2", summary)
        self.assertIn("unique_rows=2", summary)
        self.assertIn("frame_ids=IMU-1", summary)
        self.assertIn("roll_deg min=0.000000 median=0.000000 max=0.000000", summary)
        self.assertIn("pitch_deg min=0.000000 median=0.000000 max=0.000000", summary)


class CommandLineTest(unittest.TestCase):
    def test_missing_bag_returns_two_and_prints_error(self):
        main = getattr(analyzer, "main", None)
        self.assertIsNotNone(main, "main is not implemented")
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.bag"
            output = Path(directory) / "output.csv"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "--bag",
                        str(missing),
                        "--output-csv",
                        str(output),
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("error:", stderr.getvalue())
        self.assertIn("missing.bag", stderr.getvalue())

    def test_successful_injected_bag_writes_csv_and_summary(self):
        main = getattr(analyzer, "main", None)
        self.assertIsNotNone(main, "main is not implemented")
        records = [("/imu", FakeImu(10, 1), FakeTime(100))]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.csv"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--bag",
                        "input.bag",
                        "--output-csv",
                        str(output),
                    ],
                    bag_factory=bag_factory_for(records),
                )

            self.assertTrue(output.is_file())

        self.assertEqual(exit_code, 0)
        self.assertIn("unique_rows=1", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
