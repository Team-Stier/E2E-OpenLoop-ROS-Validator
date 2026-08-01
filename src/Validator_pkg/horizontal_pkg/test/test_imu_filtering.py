#!/usr/bin/env python3

import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from horizontal_pkg import imu_filtering as filtering


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


class FilterOrientationTest(unittest.TestCase):
    def irregular_signal(self):
        indices = np.arange(800, dtype=float)
        intervals = 0.025 + 0.002 * np.sin(indices * 0.31)
        elapsed = np.cumsum(intervals)
        elapsed -= elapsed[0]
        stamps = 1_785_489_005_000_000_000 + np.rint(
            elapsed * 1e9
        ).astype(np.int64)
        low = np.sin(2.0 * math.pi * 0.7 * elapsed)
        noise = 0.4 * np.sin(2.0 * math.pi * 8.0 * elapsed)
        return stamps, elapsed, low, low + noise

    def test_irregular_samples_preserve_low_frequency_and_attenuate_noise(self):
        stamps, _, low, measured = self.irregular_signal()

        result = filtering.filter_orientation(
            stamps,
            measured,
            -0.5 * measured,
            cutoff_hz=4.0,
        )

        interior = slice(50, -50)
        raw_error = rms(measured[interior] - low[interior])
        filtered_error = rms(
            result.roll_deg_filtered[interior] - low[interior]
        )
        self.assertLess(filtered_error, 0.05)
        self.assertLess(filtered_error, raw_error * 0.25)
        self.assertTrue(
            np.allclose(
                result.pitch_deg_filtered,
                -0.5 * result.roll_deg_filtered,
                atol=1e-12,
            )
        )
        self.assertAlmostEqual(result.cutoff_hz, 4.0)
        self.assertGreater(result.sample_rate_hz, 35.0)
        self.assertLess(result.sample_rate_hz, 45.0)

    def test_zero_phase_filter_does_not_shift_low_frequency_signal(self):
        stamps, _, low, measured = self.irregular_signal()

        result = filtering.filter_orientation(
            stamps,
            measured,
            measured,
            cutoff_hz=4.0,
        )

        interior = slice(50, -50)
        reference = low[interior] - np.mean(low[interior])
        filtered = result.roll_deg_filtered[interior]
        filtered = filtered - np.mean(filtered)
        correlation = np.correlate(filtered, reference, mode="full")
        lag_samples = int(np.argmax(correlation) - (len(reference) - 1))
        self.assertEqual(lag_samples, 0)

    def assert_filter_error(self, fragment, stamps, roll, pitch, cutoff=4.0):
        with self.assertRaisesRegex(filtering.FilteringError, fragment):
            filtering.filter_orientation(stamps, roll, pitch, cutoff)

    def test_non_finite_angle_is_rejected(self):
        self.assert_filter_error(
            "finite",
            [0, 25_000_000, 50_000_000, 75_000_000],
            [0.0, math.nan, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        )

    def test_non_finite_timestamp_is_rejected(self):
        self.assert_filter_error(
            "finite",
            [0.0, 25_000_000.0, math.inf, 75_000_000.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        )

    def test_duplicate_timestamp_is_rejected(self):
        self.assert_filter_error(
            "strictly increasing",
            [0, 25_000_000, 25_000_000, 50_000_000],
            [0.0] * 4,
            [0.0] * 4,
        )

    def test_reversed_timestamp_is_rejected(self):
        self.assert_filter_error(
            "strictly increasing",
            [0, 50_000_000, 25_000_000, 75_000_000],
            [0.0] * 4,
            [0.0] * 4,
        )

    def test_non_positive_cutoff_is_rejected(self):
        self.assert_filter_error(
            "positive",
            np.arange(100) * 25_000_000,
            np.zeros(100),
            np.zeros(100),
            cutoff=0.0,
        )

    def test_cutoff_at_nyquist_is_rejected(self):
        self.assert_filter_error(
            "Nyquist",
            np.arange(100) * 25_000_000,
            np.zeros(100),
            np.zeros(100),
            cutoff=20.0,
        )

    def test_too_short_input_is_rejected(self):
        self.assert_filter_error(
            "too short",
            np.arange(4) * 25_000_000,
            np.zeros(4),
            np.zeros(4),
        )

    def test_numeric_timestamp_strings_use_numeric_order(self):
        stamps = [str(index * 25_000_000) for index in range(20)]

        result = filtering.filter_orientation(
            stamps,
            np.zeros(20),
            np.zeros(20),
        )

        self.assertAlmostEqual(result.sample_rate_hz, 40.0)


class CsvPlotTest(unittest.TestCase):
    fieldnames = (
        "header_stamp_ns",
        "elapsed_s",
        "seq",
        "roll_deg",
        "pitch_deg",
    )

    def write_input(self, path, fieldnames=None, count=160):
        names = tuple(fieldnames or self.fieldnames)
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            for index in range(count):
                elapsed = index * 0.025
                row = {
                    "header_stamp_ns": str(
                        1_785_489_005_000_000_000 + index * 25_000_000
                    ),
                    "elapsed_s": repr(elapsed),
                    "seq": str(9000 + index),
                    "roll_deg": repr(
                        math.sin(2.0 * math.pi * 0.7 * elapsed)
                        + 0.2 * math.sin(2.0 * math.pi * 8.0 * elapsed)
                    ),
                    "pitch_deg": repr(
                        0.5 * math.sin(2.0 * math.pi * 0.5 * elapsed)
                    ),
                }
                writer.writerow({name: row[name] for name in names})

    def test_filter_csv_preserves_source_and_appends_results_and_png(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "filtered.csv"
            png_path = Path(directory) / "comparison.png"
            self.write_input(input_path)
            source_bytes = input_path.read_bytes()

            result = filtering.filter_csv(
                input_path,
                output_path,
                png_path,
                cutoff_hz=4.0,
            )

            self.assertEqual(input_path.read_bytes(), source_bytes)
            with input_path.open(newline="", encoding="utf-8") as stream:
                source_rows = list(csv.DictReader(stream))
            with output_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                output_fields = tuple(reader.fieldnames)
                output_rows = list(reader)
            self.assertEqual(
                output_fields,
                self.fieldnames
                + ("roll_deg_filtered", "pitch_deg_filtered"),
            )
            self.assertEqual(len(output_rows), len(source_rows))
            for source, output in zip(source_rows, output_rows):
                for field in self.fieldnames:
                    self.assertEqual(output[field], source[field])
            self.assertTrue(
                any(
                    row["roll_deg_filtered"] != row["roll_deg"]
                    for row in output_rows
                )
            )
            self.assertEqual(png_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(png_path.stat().st_size, 1000)
            self.assertEqual(result.sample_count, 160)
            self.assertAlmostEqual(result.filter_result.cutoff_hz, 4.0)

    def test_missing_required_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            self.write_input(
                input_path,
                fieldnames=(
                    "header_stamp_ns",
                    "elapsed_s",
                    "seq",
                    "roll_deg",
                ),
            )
            with self.assertRaisesRegex(filtering.FilteringError, "pitch_deg"):
                filtering.filter_csv(
                    input_path,
                    Path(directory) / "filtered.csv",
                    Path(directory) / "comparison.png",
                )

    def test_extra_csv_value_is_rejected_as_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            self.write_input(input_path)
            lines = input_path.read_text(encoding="utf-8").splitlines()
            lines[1] += ",unexpected"
            input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(filtering.FilteringError, "malformed"):
                filtering.filter_csv(
                    input_path,
                    Path(directory) / "filtered.csv",
                    Path(directory) / "comparison.png",
                )

    def test_duplicate_csv_header_is_rejected_as_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            self.write_input(input_path)
            lines = input_path.read_text(encoding="utf-8").splitlines()
            lines[0] = lines[0].replace("pitch_deg", "roll_deg")
            input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(filtering.FilteringError, "duplicate"):
                filtering.filter_csv(
                    input_path,
                    Path(directory) / "filtered.csv",
                    Path(directory) / "comparison.png",
                )

    def test_input_path_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            self.write_input(input_path)
            with self.assertRaisesRegex(filtering.FilteringError, "differ"):
                filtering.filter_csv(
                    input_path,
                    input_path,
                    Path(directory) / "comparison.png",
                )

    def test_failed_atomic_replace_preserves_old_output_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "filtered.csv"
            png_path = Path(directory) / "comparison.png"
            self.write_input(input_path)
            output_path.write_text("old output\n", encoding="utf-8")
            names_before = set(os.listdir(directory))

            with mock.patch.object(
                filtering.os,
                "replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(
                    filtering.FilteringError,
                    "failed to write",
                ):
                    filtering.filter_csv(
                        input_path,
                        output_path,
                        png_path,
                    )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "old output\n",
            )
            self.assertEqual(set(os.listdir(directory)), names_before)

    def test_csv_temporary_file_failure_is_wrapped(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            self.write_input(input_path)

            with mock.patch.object(
                filtering.tempfile,
                "NamedTemporaryFile",
                side_effect=OSError("temporary failed"),
            ):
                with self.assertRaisesRegex(
                    filtering.FilteringError,
                    "failed to write output CSV",
                ):
                    filtering.filter_csv(
                        input_path,
                        Path(directory) / "filtered.csv",
                        Path(directory) / "comparison.png",
                    )

    def test_png_temporary_file_failure_is_wrapped(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "filtered.csv"
            png_path = Path(directory) / "comparison.png"
            first_temporary = Path(directory) / ".filtered-temporary.csv"
            self.write_input(input_path)

            with mock.patch.object(
                filtering,
                "_temporary_name",
                side_effect=[first_temporary, OSError("temporary failed")],
            ):
                with self.assertRaisesRegex(
                    filtering.FilteringError,
                    "failed to write output PNG",
                ):
                    filtering.filter_csv(
                        input_path,
                        output_path,
                        png_path,
                    )

            self.assertTrue(output_path.is_file())
            self.assertFalse(first_temporary.exists())


class CommandLineTest(unittest.TestCase):
    def test_cli_writes_outputs_and_reports_summary(self):
        fixture = CsvPlotTest()
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "filtered.csv"
            png_path = Path(directory) / "comparison.png"
            fixture.write_input(input_path)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = filtering.main(
                    [
                        "--input-csv",
                        str(input_path),
                        "--output-csv",
                        str(output_path),
                        "--output-png",
                        str(png_path),
                        "--cutoff-hz",
                        "4.0",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.is_file())
            self.assertTrue(png_path.is_file())
            summary = stdout.getvalue()
            self.assertIn("samples=160", summary)
            self.assertIn("cutoff_hz=4.000000", summary)
            self.assertIn("roll_std_deg raw=", summary)
            self.assertIn("pitch_std_deg raw=", summary)

    def test_cli_expected_error_returns_two(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = filtering.main(
                [
                    "--input-csv",
                    "/missing/input.csv",
                    "--output-csv",
                    "/tmp/filtered.csv",
                    "--output-png",
                    "/tmp/comparison.png",
                ]
            )

        self.assertEqual(exit_code, 2)
        self.assertIn("error:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
