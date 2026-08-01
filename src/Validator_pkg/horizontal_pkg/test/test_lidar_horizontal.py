#!/usr/bin/env python3

import math
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
