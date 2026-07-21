"""
ROS binary extension 없이 quaternion spherical interpolation을 계산한다.
입력: xyzw quaternion 두 개와 보간 비율.
출력(pub): 없음. 정규화한 xyzw quaternion 배열을 반환한다.
구독(sub): 없음.
"""

from typing import Sequence

import numpy as np


def quaternion_slerp(
    q0: Sequence[float], q1: Sequence[float], fraction: float
) -> np.ndarray:
    """두 xyzw quaternion 사이의 최단 구면 선형 보간값을 반환한다.

    입력: 0이 아닌 시작·끝 quaternion과 보간 비율.
    반환: 단위 길이의 NumPy xyzw quaternion.
    """
    start = np.asarray(q0, dtype=np.float64)
    end = np.asarray(q1, dtype=np.float64)
    start_norm = float(np.linalg.norm(start))
    end_norm = float(np.linalg.norm(end))
    if start_norm == 0.0 or end_norm == 0.0:
        raise ValueError("quaternions must be nonzero")

    start = start / start_norm
    end = end / end_norm
    if fraction == 0.0:
        return start
    if fraction == 1.0:
        return end

    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        result = start + fraction * (end - start)
        return result / np.linalg.norm(result)

    angle = float(np.arccos(dot))
    inverse_sine = 1.0 / np.sin(angle)
    start_weight = np.sin((1.0 - fraction) * angle) * inverse_sine
    end_weight = np.sin(fraction * angle) * inverse_sine
    return start_weight * start + end_weight * end
