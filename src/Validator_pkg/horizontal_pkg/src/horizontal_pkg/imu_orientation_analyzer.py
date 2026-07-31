"""Extract roll and pitch from IMU orientation quaternions in a ROS bag."""

import math
from dataclasses import dataclass


class AnalyzerError(RuntimeError):
    """Base error for rejected analyzer input or output."""


class InvalidQuaternionError(AnalyzerError):
    """Raised when an orientation quaternion cannot represent a rotation."""


@dataclass(frozen=True)
class OrientationAngles:
    """Roll and pitch obtained from one normalized quaternion."""

    quaternion_norm: float
    roll_rad: float
    pitch_rad: float
    normalization_warning: bool


def quaternion_to_roll_pitch(x, y, z, w):
    """Convert a ROS-order quaternion into roll and pitch radians."""
    values = tuple(float(value) for value in (x, y, z, w))
    if not all(math.isfinite(value) for value in values):
        raise InvalidQuaternionError(
            "quaternion contains a non-finite component"
        )

    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-6:
        raise InvalidQuaternionError(
            "quaternion norm is smaller than 1e-6"
        )

    x, y, z, w = (value / norm for value in values)
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    sin_pitch = max(
        -1.0,
        min(1.0, 2.0 * (w * y - z * x)),
    )
    pitch = math.asin(sin_pitch)

    return OrientationAngles(
        quaternion_norm=norm,
        roll_rad=roll,
        pitch_rad=pitch,
        normalization_warning=abs(norm - 1.0) > 1e-3,
    )
