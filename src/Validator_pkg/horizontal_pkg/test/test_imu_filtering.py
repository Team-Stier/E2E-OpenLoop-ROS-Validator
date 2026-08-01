#!/usr/bin/env python3

import math
import unittest

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


if __name__ == "__main__":
    unittest.main()
