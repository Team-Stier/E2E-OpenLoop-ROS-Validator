#!/usr/bin/env python3

from contextlib import redirect_stderr, redirect_stdout
import io
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from horizontal_pkg import lidar_horizontal as horizontal


class GeometryTest(unittest.TestCase):
    def assert_vector(self, actual, expected):
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_positive_roll_rotates_left_axis_up(self):
        rotation = horizontal.level_rotation(
            math.pi / 2.0,
            0.0,
            np.eye(3),
        )

        self.assert_vector(rotation @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])

    def test_positive_pitch_rotates_forward_axis_down(self):
        rotation = horizontal.level_rotation(
            0.0,
            math.pi / 2.0,
            np.eye(3),
        )

        self.assert_vector(rotation @ [1.0, 0.0, 0.0], [0.0, 0.0, -1.0])

    def test_combined_leveling_uses_pitch_after_roll(self):
        rotation = horizontal.level_rotation(
            math.pi / 2.0,
            math.pi / 2.0,
            np.eye(3),
        )

        self.assert_vector(rotation @ [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])

    def test_sensor_relative_rotation_is_applied_first(self):
        lidar_to_imu = horizontal.rotation_matrix_xyz(0.0, 0.0, math.pi / 2.0)

        rotation = horizontal.level_rotation(0.0, 0.0, lidar_to_imu)

        self.assert_vector(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    def test_non_rotation_extrinsic_is_rejected(self):
        invalid = np.diag([1.0, 1.0, 2.0])
        with self.assertRaisesRegex(horizontal.HorizontalizationError, "rotation"):
            horizontal.level_rotation(0.0, 0.0, invalid)

    def test_combined_attitude_levels_synthetic_plane(self):
        roll, pitch = 0.12, -0.08
        rotation = horizontal.level_rotation(roll, pitch, np.eye(3))
        level_points = np.array([[x, y, 0.0] for x in (-2.0, 0.0, 3.0) for y in (-1.0, 2.0)])
        tilted_points = level_points @ rotation
        leveled = tilted_points @ rotation.T
        np.testing.assert_allclose(leveled[:, 2], 0.0, atol=1e-12)


class AttitudeTimelineTest(unittest.TestCase):
    def timeline(self):
        return horizontal.AttitudeTimeline(
            stamps_ns=(1_000_000_000, 1_050_000_000, 1_100_000_000),
            roll_rad=(0.0, 0.1, 0.2),
            pitch_rad=(0.0, -0.2, -0.4),
        )

    def test_exact_stamp_returns_exact_angles(self):
        self.assertEqual(
            self.timeline().interpolate(1_050_000_000, 0.10),
            (0.1, -0.2),
        )

    def test_bracketed_stamp_is_linearly_interpolated(self):
        roll, pitch = self.timeline().interpolate(1_075_000_000, 0.10)

        self.assertAlmostEqual(roll, 0.15)
        self.assertAlmostEqual(pitch, -0.3)

    def test_stamp_outside_timeline_is_not_extrapolated(self):
        self.assertIsNone(self.timeline().interpolate(999_999_999, 0.10))
        self.assertIsNone(self.timeline().interpolate(1_100_000_001, 0.10))

    def test_bracket_span_over_limit_is_rejected(self):
        timeline = horizontal.AttitudeTimeline(
            stamps_ns=(0, 200_000_000),
            roll_rad=(0.0, 0.2),
            pitch_rad=(0.0, 0.4),
        )

        self.assertIsNone(timeline.interpolate(100_000_000, 0.10))

    def test_one_sided_age_over_limit_is_rejected(self):
        timeline = horizontal.AttitudeTimeline(
            stamps_ns=(0, 180_000_000),
            roll_rad=(0.0, 0.2),
            pitch_rad=(0.0, 0.4),
        )

        self.assertIsNone(timeline.interpolate(10_000_000, 0.15))

    def test_non_increasing_stamps_are_rejected(self):
        with self.assertRaisesRegex(
            horizontal.HorizontalizationError,
            "strictly increasing",
        ):
            horizontal.AttitudeTimeline(
                stamps_ns=(10, 10),
                roll_rad=(0.0, 0.1),
                pitch_rad=(0.0, 0.1),
            )

    def test_non_positive_max_gap_is_rejected(self):
        with self.assertRaisesRegex(horizontal.HorizontalizationError, "positive"):
            self.timeline().interpolate(1_050_000_000, 0.0)


def transform(parent, child, translation, quaternion=(0.0, 0.0, 0.0, 1.0)):
    message = TransformStamped()
    message.header.frame_id = parent
    message.child_frame_id = child
    message.transform.translation.x = translation[0]
    message.transform.translation.y = translation[1]
    message.transform.translation.z = translation[2]
    message.transform.rotation.x = quaternion[0]
    message.transform.rotation.y = quaternion[1]
    message.transform.rotation.z = quaternion[2]
    message.transform.rotation.w = quaternion[3]
    return message


class ExtrinsicsTest(unittest.TestCase):
    def messages(self):
        messages = []
        for offset in (-1e-5, 0.0, 1e-5):
            messages.append(
                TFMessage(
                    transforms=[
                        transform(
                            "/base_link",
                            "IMU-1",
                            (1.5 + offset, 0.0, 0.5),
                        ),
                        transform(
                            "/base_link",
                            "Lidar3D-4",
                            (1.5, 0.0, 1.25),
                        ),
                    ]
                )
            )
        return messages

    def test_bag_transforms_produce_vertical_sensor_offset(self):
        result = horizontal.extract_extrinsics(
            self.messages(),
            base_frame="base_link",
            imu_frame="IMU-1",
            lidar_frame="Lidar3D-4",
        )

        np.testing.assert_allclose(
            result.T_base_from_imu.translation_m,
            [1.5, 0.0, 0.5],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.T_base_from_lidar.translation_m,
            [1.5, 0.0, 1.25],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.translation_imu_from_lidar_m,
            [0.0, 0.0, 0.75],
            atol=1e-12,
        )
        np.testing.assert_allclose(result.R_imu_from_lidar, np.eye(3), atol=1e-12)

    def test_missing_sensor_transform_is_rejected(self):
        messages = [
            TFMessage(
                transforms=[transform("/base_link", "IMU-1", (1.5, 0.0, 0.5))]
            )
        ]
        with self.assertRaisesRegex(horizontal.HorizontalizationError, "Lidar3D-4"):
            horizontal.extract_extrinsics(
                messages,
                "base_link",
                "IMU-1",
                "Lidar3D-4",
            )

    def test_unstable_translation_is_rejected(self):
        messages = self.messages()
        messages.append(
            TFMessage(
                transforms=[
                    transform("/base_link", "IMU-1", (1.6, 0.0, 0.5)),
                    transform("/base_link", "Lidar3D-4", (1.5, 0.0, 1.25)),
                ]
            )
        )
        with self.assertRaisesRegex(horizontal.HorizontalizationError, "unstable"):
            horizontal.extract_extrinsics(
                messages,
                "base_link",
                "IMU-1",
                "Lidar3D-4",
            )

    def test_non_finite_transform_is_rejected(self):
        messages = self.messages()
        messages[0].transforms[0].transform.translation.x = math.nan
        with self.assertRaisesRegex(horizontal.HorizontalizationError, "finite"):
            horizontal.extract_extrinsics(
                messages,
                "base_link",
                "IMU-1",
                "Lidar3D-4",
            )


def imu_message(stamp_ns, roll_rad=0.0, pitch_rad=0.0, valid=True):
    message = Imu()
    message.header.stamp.secs = stamp_ns // 1_000_000_000
    message.header.stamp.nsecs = stamp_ns % 1_000_000_000
    message.header.frame_id = "IMU-1"
    if valid:
        cr, sr = math.cos(roll_rad / 2.0), math.sin(roll_rad / 2.0)
        cp, sp = math.cos(pitch_rad / 2.0), math.sin(pitch_rad / 2.0)
        message.orientation.x = sr * cp
        message.orientation.y = cr * sp
        message.orientation.z = -sr * sp
        message.orientation.w = cr * cp
    return message


class ImuPreparationTest(unittest.TestCase):
    def test_duplicate_stamp_keeps_last_and_invalid_is_counted(self):
        messages = [
            imu_message((index + 1) * 25_000_000, roll_rad=index * 0.001)
            for index in range(24)
        ]
        messages.append(imu_message(11 * 25_000_000, roll_rad=0.5))
        messages.append(imu_message(700_000_000, valid=False))

        timeline, stats = horizontal.prepare_attitude(messages, cutoff_hz=4.0)

        self.assertEqual(stats.total_messages, 26)
        self.assertEqual(stats.valid_unique_messages, 24)
        self.assertEqual(stats.duplicate_replacements, 1)
        self.assertEqual(stats.invalid_quaternion_count, 1)
        self.assertEqual(len(timeline.stamps_ns), 24)
        self.assertEqual(timeline.stamps_ns[10], 275_000_000)

    def test_zero_stamp_is_rejected_and_counted(self):
        messages = [
            imu_message((index + 1) * 25_000_000) for index in range(20)
        ]
        messages.append(imu_message(0))

        _, stats = horizontal.prepare_attitude(messages, cutoff_hz=4.0)

        self.assertEqual(stats.zero_stamp_count, 1)

    def test_long_dropout_is_filtered_as_separate_segments(self):
        first = [imu_message((i + 1) * 25_000_000, roll_rad=0.0) for i in range(20)]
        second = [imu_message(2_000_000_000 + i * 25_000_000, roll_rad=1.0) for i in range(20)]
        timeline, _ = horizontal.prepare_attitude(first + second, max_imu_gap_s=0.10)
        self.assertAlmostEqual(timeline.roll_rad[19], 0.0, places=6)
        self.assertAlmostEqual(timeline.roll_rad[20], 1.0, places=6)
        self.assertIsNone(timeline.interpolate(1_000_000_000, 0.10))

    def test_unexpected_imu_frame_is_rejected(self):
        messages = [imu_message((i + 1) * 25_000_000) for i in range(20)]
        messages[5].header.frame_id = "other_imu"
        with self.assertRaisesRegex(horizontal.HorizontalizationError, "IMU frame"):
            horizontal.prepare_attitude(messages, expected_frame="IMU-1")


def point_field(name, offset, datatype, count=1):
    return PointField(name=name, offset=offset, datatype=datatype, count=count)


def cloud_message(bigendian=False, bad_row_step=False):
    message = PointCloud2()
    message.header.frame_id = "Lidar3D-4"
    message.header.stamp.secs = 12
    message.header.stamp.nsecs = 34
    message.height = 2
    message.width = 2
    message.fields = [
        point_field("x", 0, PointField.FLOAT32),
        point_field("y", 4, PointField.FLOAT32),
        point_field("z", 8, PointField.FLOAT32),
        point_field("intensity", 12, PointField.FLOAT32),
        point_field("ring", 16, PointField.UINT16),
    ]
    message.is_bigendian = bigendian
    message.point_step = 20
    message.row_step = 39 if bad_row_step else 44
    message.is_dense = False
    endian = ">" if bigendian else "<"
    payload = bytearray(message.row_step * message.height)
    values = [
        (1.0, 2.0, 3.0, 10.0, 7),
        (4.0, 5.0, 6.0, 20.0, 8),
        (math.nan, 8.0, 9.0, 30.0, 9),
        (-1.0, -2.0, -3.0, 40.0, 10),
    ]
    for index, value in enumerate(values):
        row, column = divmod(index, message.width)
        offset = row * message.row_step + column * message.point_step
        struct.pack_into(endian + "ffffH", payload, offset, *value)
        payload[offset + 18 : offset + 20] = b"XY"
    if not bad_row_step:
        payload[40:44] = b"PAD1"
        payload[84:88] = b"PAD2"
    message.data = bytes(payload)
    return message


class PointCloudTest(unittest.TestCase):
    def test_xyz_rotates_while_layout_and_non_xyz_bytes_are_preserved(self):
        source = cloud_message()
        rotation = horizontal.rotation_matrix_xyz(math.pi / 2.0, 0.0)

        output = horizontal.horizontalize_cloud(
            source,
            rotation,
            "Lidar3D-4-horizontal",
        )

        self.assertEqual(output.header.stamp, source.header.stamp)
        self.assertEqual(output.header.frame_id, "Lidar3D-4-horizontal")
        self.assertEqual(output.height, source.height)
        self.assertEqual(output.width, source.width)
        self.assertEqual(output.fields, source.fields)
        self.assertEqual(output.point_step, source.point_step)
        self.assertEqual(output.row_step, source.row_step)
        for index, expected in enumerate(
            ((1.0, -3.0, 2.0), (4.0, -6.0, 5.0), None, (-1.0, 3.0, -2.0))
        ):
            row, column = divmod(index, source.width)
            offset = row * source.row_step + column * source.point_step
            actual = struct.unpack_from("<fff", output.data, offset)
            if expected is None:
                self.assertTrue(all(math.isnan(value) for value in actual))
            else:
                np.testing.assert_allclose(actual, expected, atol=1e-6)
            self.assertEqual(output.data[offset + 12 : offset + 20], source.data[offset + 12 : offset + 20])
        self.assertEqual(output.data[40:44], b"PAD1")
        self.assertEqual(output.data[84:88], b"PAD2")

    def test_big_endian_cloud_is_supported(self):
        output = horizontal.horizontalize_cloud(
            cloud_message(bigendian=True),
            np.eye(3),
            "horizontal",
        )
        self.assertEqual(struct.unpack_from(">fff", output.data, 0), (1.0, 2.0, 3.0))

    def test_non_float_xyz_is_rejected(self):
        message = cloud_message()
        message.fields[0].datatype = PointField.FLOAT64
        with self.assertRaisesRegex(horizontal.HorizontalizationError, "FLOAT32"):
            horizontal.horizontalize_cloud(message, np.eye(3), "horizontal")

    def test_overlapping_xyz_and_other_field_is_rejected(self):
        message = cloud_message()
        message.fields[-1].offset = 10
        with self.assertRaisesRegex(horizontal.HorizontalizationError, "overlap"):
            horizontal.horizontalize_cloud(message, np.eye(3), "horizontal")

    def test_inconsistent_row_step_is_rejected(self):
        with self.assertRaisesRegex(horizontal.HorizontalizationError, "row_step"):
            horizontal.horizontalize_cloud(
                cloud_message(bad_row_step=True),
                np.eye(3),
                "horizontal",
            )

    def test_payload_length_mismatch_is_rejected(self):
        message = cloud_message()
        message.data = message.data[:-1]
        with self.assertRaisesRegex(horizontal.HorizontalizationError, "payload"):
            horizontal.horizontalize_cloud(message, np.eye(3), "horizontal")


class FakeBag:
    def __init__(self, records=None):
        self.records = list(records or [])
        self.writes = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        self.closed = True

    def read_messages(self, topics=None, return_connection_header=False):
        selected = None if topics is None else set(topics)
        for record in self.records:
            if selected is None or record[0] in selected:
                if return_connection_header:
                    yield record + ({"callerid": "original", "latching": "1"},)
                else:
                    yield record

    def write(self, topic, message, timestamp, connection_header=None):
        self.writes.append((topic, message, timestamp, connection_header))


class FakeBagFactory:
    def __init__(self, records):
        self.records = records
        self.read_bags = []
        self.write_bags = []
        self.calls = []

    def __call__(self, path, mode, compression=None):
        self.calls.append((path, mode, compression))
        if mode == "r":
            bag = FakeBag(self.records)
            self.read_bags.append(bag)
            return bag
        bag = FakeBag()
        self.write_bags.append(bag)
        return bag


def set_cloud_stamp(message, stamp_ns):
    message.header.stamp.secs = stamp_ns // 1_000_000_000
    message.header.stamp.nsecs = stamp_ns % 1_000_000_000
    return message


class BagPipelineTest(unittest.TestCase):
    def records(self, include_tf=True, aligned=True):
        records = []
        if include_tf:
            for index, message in enumerate(ExtrinsicsTest().messages()):
                records.append(("/tf", message, rospy.Time.from_sec(0.01 * index)))
        for index in range(24):
            stamp_ns = (index + 1) * 25_000_000
            records.append(
                (
                    "/imu",
                    imu_message(stamp_ns, roll_rad=0.05, pitch_rad=-0.02),
                    rospy.Time.from_sec(0.1 + index * 0.025),
                )
            )
        lidar_stamp = 300_000_000 if aligned else 2_000_000_000
        records.append(
            (
                "/lidar3D",
                set_cloud_stamp(cloud_message(), lidar_stamp),
                rospy.Time.from_sec(1.0),
            )
        )
        records.append(("/other", String(data="preserve"), rospy.Time.from_sec(1.1)))
        return records

    def test_pipeline_copies_records_and_inserts_horizontal_cloud(self):
        records = self.records()
        factory = FakeBagFactory(records)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "horizontal.bag"

            stats = horizontal.process_bag(
                "/virtual/input.bag",
                output_path,
                bag_factory=factory,
            )

            writes = factory.write_bags[0].writes
            self.assertEqual(len(factory.read_bags), 2)
            self.assertTrue(all(bag.closed for bag in factory.read_bags))
            self.assertTrue(factory.write_bags[0].closed)
            self.assertEqual(len(writes), len(records) + 1)
            raw_index = next(i for i, row in enumerate(writes) if row[0] == "/lidar3D")
            derived = writes[raw_index + 1]
            self.assertEqual(derived[0], "/lidar3D/horizontal")
            self.assertEqual(derived[2], writes[raw_index][2])
            self.assertIsNone(derived[3])
            self.assertEqual(writes[raw_index][3]["latching"], "1")
            self.assertEqual(
                derived[1].header.stamp,
                writes[raw_index][1].header.stamp,
            )
            self.assertEqual(derived[1].header.frame_id, "Lidar3D-4-horizontal")
            self.assertEqual(stats.lidar_input_count, 1)
            self.assertEqual(stats.lidar_written_count, 1)
            self.assertEqual(stats.lidar_skipped_alignment_count, 0)
            self.assertTrue(output_path.exists())
            write_calls = [call for call in factory.calls if call[1] == "w"]
            self.assertEqual(write_calls[0][2], "lz4")

    def test_topic_collisions_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for topic in ("/imu", "/tf", "/lidar3D"):
                with self.assertRaisesRegex(horizontal.HorizontalizationError, "output topic"):
                    horizontal.process_bag("/virtual/input.bag", Path(directory) / "x.bag", output_topic=topic, bag_factory=FakeBagFactory(self.records()))
            records = self.records() + [("/lidar3D/horizontal", cloud_message(), rospy.Time.from_sec(2.0))]
            with self.assertRaisesRegex(horizontal.HorizontalizationError, "already contains"):
                horizontal.process_bag("/virtual/input.bag", Path(directory) / "x.bag", bag_factory=FakeBagFactory(records))

    def test_unexpected_lidar_frame_is_rejected(self):
        records = self.records()
        next(row[1] for row in records if row[0] == "/lidar3D").header.frame_id = "other_lidar"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(horizontal.HorizontalizationError, "LiDAR frame"):
                horizontal.process_bag("/virtual/input.bag", Path(directory) / "x.bag", bag_factory=FakeBagFactory(records))

    def test_unaligned_cloud_is_skipped_and_zero_output_fails(self):
        factory = FakeBagFactory(self.records(aligned=False))
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "horizontal.bag"
            with self.assertRaisesRegex(horizontal.HorizontalizationError, "zero"):
                horizontal.process_bag(
                    "/virtual/input.bag",
                    output_path,
                    bag_factory=factory,
                )
            self.assertFalse(output_path.exists())

    def test_missing_tf_is_rejected_before_output_open(self):
        factory = FakeBagFactory(self.records(include_tf=False))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(horizontal.HorizontalizationError, "TF"):
                horizontal.process_bag(
                    "/virtual/input.bag",
                    Path(directory) / "horizontal.bag",
                    bag_factory=factory,
                )
            self.assertEqual(factory.write_bags, [])

    def test_input_and_output_paths_must_differ(self):
        factory = FakeBagFactory(self.records())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same.bag"
            with self.assertRaisesRegex(horizontal.HorizontalizationError, "differ"):
                horizontal.process_bag(path, path, bag_factory=factory)

    def test_failed_replace_preserves_destination_and_removes_temporary(self):
        factory = FakeBagFactory(self.records())
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "horizontal.bag"
            output_path.write_bytes(b"old")
            before = set(os.listdir(directory))
            with mock.patch.object(
                horizontal.os,
                "replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(
                    horizontal.HorizontalizationError,
                    "failed to write output bag",
                ):
                    horizontal.process_bag(
                        "/virtual/input.bag",
                        output_path,
                        bag_factory=factory,
                    )
            self.assertEqual(output_path.read_bytes(), b"old")
            self.assertEqual(set(os.listdir(directory)), before)


class CommandLineTest(unittest.TestCase):
    def test_cli_success_reports_counts_and_extrinsics(self):
        factory = FakeBagFactory(BagPipelineTest().records())
        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = horizontal.main(
                    [
                        "--input-bag",
                        "/virtual/input.bag",
                        "--output-bag",
                        str(Path(directory) / "horizontal.bag"),
                    ],
                    bag_factory=factory,
                )
            self.assertEqual(exit_code, 0)
            summary = stdout.getvalue()
            self.assertIn("lidar_input=1 written=1 skipped_alignment=0", summary)
            self.assertIn("imu_from_lidar_translation_m=[0.0, 0.0, 0.75]", summary)

    def test_cli_expected_error_returns_two(self):
        factory = FakeBagFactory(BagPipelineTest().records(include_tf=False))
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stderr(stderr):
                exit_code = horizontal.main(
                    [
                        "--input-bag",
                        "/virtual/input.bag",
                        "--output-bag",
                        str(Path(directory) / "horizontal.bag"),
                    ],
                    bag_factory=factory,
                )
        self.assertEqual(exit_code, 2)
        self.assertIn("error:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
