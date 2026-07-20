"""Convert source-model ego trajectories to ROS paths in the LiDAR frame."""

from __future__ import annotations

import math
from typing import Any, Sequence, Tuple

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


_OUTPUT_FRAME = "Lidar3D-4"


def _as_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("trajectory must be a numeric array") from error
    while result.ndim > 2 and result.shape[0] == 1:
        result = result[0]
    return result


def _rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    if len(quaternion) != 4:
        raise ValueError("T_base_from_lidar quaternion must have length 4")
    x, y, z, w = (float(item) for item in quaternion)
    norm = x * x + y * y + z * z + w * w
    if not math.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError("T_base_from_lidar quaternion must be finite and nonzero")
    scale = 2.0 / norm
    return np.asarray(
        [
            [
                1.0 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
            ],
            [
                scale * (x * y + z * w),
                1.0 - scale * (x * x + z * z),
                scale * (y * z - x * w),
            ],
            [
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1.0 - scale * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _normalized_quaternion(quaternion: Sequence[float]) -> Tuple[float, ...]:
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("T_base_from_lidar quaternion must have four finite values")
    norm = float(np.linalg.norm(values))
    if norm <= np.finfo(float).eps:
        raise ValueError("T_base_from_lidar quaternion must be finite and nonzero")
    return tuple(float(item) for item in values / norm)


def _quaternion_product(
    left: Sequence[float], right: Sequence[float]
) -> Tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


class PathMessageBuilder:
    def __init__(self, t_base_from_lidar: dict[str, Any]) -> None:
        try:
            parent_frame = str(t_base_from_lidar["parent_frame_id"])
            child_frame = str(t_base_from_lidar["child_frame_id"])
            translation = np.asarray(
                t_base_from_lidar["translation_m"], dtype=np.float64
            )
            quaternion = _normalized_quaternion(
                t_base_from_lidar["quaternion_xyzw"]
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ValueError):
                raise
            raise ValueError(
                "T_base_from_lidar requires frame ids, translation, and quaternion"
            ) from error
        if parent_frame != "base_link" or child_frame != _OUTPUT_FRAME:
            raise ValueError(
                "T_base_from_lidar must map base_link to Lidar3D-4"
            )
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError(
                "T_base_from_lidar translation must have three finite values"
            )
        self._translation = translation
        self._rotation = _rotation_matrix(quaternion)
        self._q_lidar_from_base = (
            -quaternion[0],
            -quaternion[1],
            -quaternion[2],
            quaternion[3],
        )

    def build(
        self, trajectory: Any, coordinate_convention: str, stamp: Any
    ) -> Path:
        convention = str(coordinate_convention).upper()
        if convention.startswith("CARLA"):
            carla = True
        elif convention.startswith("NAVSIM"):
            carla = False
        else:
            raise ValueError(
                "unsupported coordinate convention: {}".format(
                    coordinate_convention
                )
            )

        points = _as_numpy(trajectory)
        if points.size == 0:
            raise ValueError("trajectory must not be empty")
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("trajectory rows must contain two or three values")
        if not np.isfinite(points).all():
            raise ValueError("trajectory values must be finite")

        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = _OUTPUT_FRAME
        for row in points:
            base_point = np.asarray(
                [row[0], -row[1] if carla else row[1], 0.0],
                dtype=np.float64,
            )
            lidar_point = self._rotation.T @ (base_point - self._translation)

            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = _OUTPUT_FRAME
            pose.pose.position.x = float(lidar_point[0])
            pose.pose.position.y = float(lidar_point[1])
            pose.pose.position.z = float(lidar_point[2])
            if points.shape[1] == 3:
                base_yaw = -float(row[2]) if carla else float(row[2])
                base_heading = (
                    0.0,
                    0.0,
                    math.sin(base_yaw / 2.0),
                    math.cos(base_yaw / 2.0),
                )
                orientation = _quaternion_product(
                    self._q_lidar_from_base, base_heading
                )
                norm = math.sqrt(sum(item * item for item in orientation))
                orientation = tuple(item / norm for item in orientation)
                (
                    pose.pose.orientation.x,
                    pose.pose.orientation.y,
                    pose.pose.orientation.z,
                    pose.pose.orientation.w,
                ) = orientation
            else:
                pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path
