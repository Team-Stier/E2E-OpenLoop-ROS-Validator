"""
모델 ego trajectory를 Lidar3D-4 frame의 ROS Path로 변환한다.
입력: 모델 trajectory, profile T_base_from_lidar, reference stamp.
출력(pub): 없음. nav_msgs/Path를 반환.
구독(sub): 없음.
"""

from __future__ import annotations

import math
from typing import Any, Sequence, Tuple

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


_OUTPUT_FRAME = "Lidar3D-4"


def _as_numpy(value: Any) -> np.ndarray:
    """모델 tensor·목록을 검증된 2차원 numpy trajectory로 바꾼다.

    입력: tensor, numpy 배열 또는 수치 sequence.
    반환: batch 차원이 제거된 float64 numpy 배열.
    """
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
    """정규화 가능 여부를 검증해 quaternion의 3×3 회전 행렬을 만든다.

    입력: ``[x, y, z, w]`` quaternion.
    반환: Lidar와 base frame 변환에 쓰는 float64 회전 행렬.
    """
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
    """유한한 quaternion을 단위 quaternion으로 정규화한다.

    입력: ``[x, y, z, w]`` quaternion.
    반환: 길이가 1인 ``(x, y, z, w)`` tuple.
    """
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
    """두 ``xyzw`` quaternion을 Hamilton 곱으로 합성한다.

    입력: 좌·우 회전을 나타내는 quaternion 두 개.
    반환: 왼쪽 회전 뒤 오른쪽 회전을 적용한 quaternion.
    """
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


class PathMessageBuilder:
    """모델의 ego trajectory를 ``Lidar3D-4`` 기준 ``nav_msgs/Path``로 바꾼다.

    입력: profile의 ``T_base_from_lidar``와 모델 trajectory.
    반환: publisher가 그대로 발행할 stamp가 보존된 ``Path``.
    """

    def __init__(self, t_base_from_lidar: dict[str, Any]) -> None:
        """정적 base-to-LiDAR 변환을 검증하고 path 변환 상태를 준비한다.

        입력: ``base_link``에서 ``Lidar3D-4``로 향하는 profile transform.
        부작용: translation, rotation, orientation 변환용 quaternion을 보관한다.
        """
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
        """모델 좌표계의 trajectory를 LiDAR frame의 ROS Path로 직렬화한다.

        입력: ``N×2`` 또는 ``N×3`` trajectory, CARLA/NAVSIM 좌표계, 기준 stamp.
        반환: 위치와 선택적 heading을 가진 ``nav_msgs/Path``.
        """
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
