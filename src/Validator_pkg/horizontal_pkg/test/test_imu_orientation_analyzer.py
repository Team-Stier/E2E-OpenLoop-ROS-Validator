#!/usr/bin/env python3

import math
import unittest

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


if __name__ == "__main__":
    unittest.main()
