"""Create completely roll/pitch-leveled LiDAR clouds in an offline ROS bag."""

from bisect import bisect_left
from dataclasses import dataclass
import math

import numpy as np


class HorizontalizationError(RuntimeError):
    """Raised when offline horizontalization cannot satisfy its contract."""


def rotation_matrix_xyz(roll_rad, pitch_rad, yaw_rad=0.0):
    """Return ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    angles = tuple(float(value) for value in (roll_rad, pitch_rad, yaw_rad))
    if not all(math.isfinite(value) for value in angles):
        raise HorizontalizationError("rotation angles must be finite")
    roll, pitch, yaw = angles
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


def _validated_rotation(matrix):
    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise HorizontalizationError("sensor extrinsic must be a finite rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise HorizontalizationError("sensor extrinsic must be a rotation matrix")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8):
        raise HorizontalizationError("sensor extrinsic must be a proper rotation")
    return rotation


def level_rotation(roll_rad, pitch_rad, R_imu_from_lidar):
    """Return the LiDAR-to-yaw-level rotation for filtered IMU attitude."""
    lidar_to_imu = _validated_rotation(R_imu_from_lidar)
    return rotation_matrix_xyz(roll_rad, pitch_rad) @ lidar_to_imu


@dataclass(frozen=True)
class AttitudeTimeline:
    """Filtered roll/pitch values indexed by strictly increasing sensor stamps."""

    stamps_ns: tuple
    roll_rad: tuple
    pitch_rad: tuple

    def __post_init__(self):
        if not (
            len(self.stamps_ns) == len(self.roll_rad) == len(self.pitch_rad)
        ):
            raise HorizontalizationError("attitude timeline lengths must match")
        if not self.stamps_ns:
            raise HorizontalizationError("attitude timeline must not be empty")
        if any(
            current <= previous
            for previous, current in zip(self.stamps_ns, self.stamps_ns[1:])
        ):
            raise HorizontalizationError(
                "attitude stamps must be strictly increasing"
            )
        values = tuple(self.roll_rad) + tuple(self.pitch_rad)
        if not all(math.isfinite(float(value)) for value in values):
            raise HorizontalizationError("attitude angles must be finite")

    def interpolate(self, stamp_ns, max_gap_s):
        """Interpolate angles at one stamp or return None when alignment is unsafe."""
        try:
            stamp = int(stamp_ns)
            max_gap = float(max_gap_s)
        except (TypeError, ValueError) as error:
            raise HorizontalizationError("timestamp and max gap must be numeric") from error
        if not math.isfinite(max_gap) or max_gap <= 0.0:
            raise HorizontalizationError("max IMU gap must be positive and finite")

        index = bisect_left(self.stamps_ns, stamp)
        if index < len(self.stamps_ns) and self.stamps_ns[index] == stamp:
            return float(self.roll_rad[index]), float(self.pitch_rad[index])
        if index == 0 or index == len(self.stamps_ns):
            return None

        before = int(self.stamps_ns[index - 1])
        after = int(self.stamps_ns[index])
        limit_ns = max_gap * 1e9
        if (
            after - before > limit_ns
            or stamp - before > limit_ns
            or after - stamp > limit_ns
        ):
            return None
        fraction = (stamp - before) / float(after - before)
        roll = float(self.roll_rad[index - 1]) + fraction * (
            float(self.roll_rad[index]) - float(self.roll_rad[index - 1])
        )
        pitch = float(self.pitch_rad[index - 1]) + fraction * (
            float(self.pitch_rad[index]) - float(self.pitch_rad[index - 1])
        )
        return roll, pitch
