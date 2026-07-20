"""
concrete model runtime이 공유하는 결과, 오류, 좌표·checkpoint 계약 유틸리티.
입력: InferenceFrame 입력, profile static transform, checkpoint state dict.
출력(pub): 없음. ModelPrediction 또는 명시적 오류를 반환.
구독(sub): 없음.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, Tuple

import numpy as np


class ModelRuntimeError(RuntimeError):
    """선택된 source/checkpoint runtime을 load하거나 호출할 수 없음을 나타낸다."""


class SourceRuntimeError(ModelRuntimeError):
    """pinned source 또는 checkpoint가 실행 계약을 위반했음을 나타낸다."""


class ModelInputError(ValueError):
    """inference frame이 선택 source model의 input 계약을 위반했음을 나타낸다."""


@dataclass(frozen=True)
class ModelPrediction:
    """어떤 concrete model도 engine에 돌려주는 공통 추론 결과다.

    입력: source-native trajectory와 이를 만든 source symbol.
    반환: engine이 Path로 변환할 불변 prediction.
    """

    trajectory: Any
    source_symbol: str


def require_inputs(model_id: str, frame: Any, names: Iterable[str]) -> Tuple[Any, ...]:
    """model이 선언한 모든 input이 causal frame에 있는지 확인한다.

    입력: model id, ``InferenceFrame``, 필요한 input key 순서.
    반환: 요청 순서를 유지한 input value tuple.
    """
    missing = [name for name in names if name not in frame.inputs]
    if missing:
        raise ModelInputError(
            f"model {model_id} missing input(s): {', '.join(missing)}"
        )
    return tuple(frame.inputs[name] for name in names)


def require_status(model_id: str, driving_feature: Any, length: int) -> list[float]:
    """driving feature의 status vector 길이를 모델 계약과 대조한다.

    입력: model id, driving feature, 요구 status 길이.
    반환: 검증된 status feature list.
    """
    try:
        status = list(driving_feature.status_feature)
    except AttributeError as error:
        raise ModelInputError(
            f"model {model_id} driving_feature missing status_feature"
        ) from error
    if len(status) != length:
        raise ModelInputError(
            f"model {model_id} status_feature requires length {length}, got {len(status)}"
        )
    return status


def prepend_source_path(path: Path) -> None:
    """pinned source checkout을 import 탐색 순서의 맨 앞에 둔다.

    입력: 원본 model source checkout 경로.
    부작용: 아직 없을 때만 ``sys.path`` 첫 항목으로 삽입한다.
    """
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def require_module_from_checkout(module: Any, checkout: Path, symbol: str) -> None:
    """import된 symbol이 profile이 고정한 checkout 내부에 있음을 검증한다.

    입력: import module, checkout directory, 사람이 읽을 symbol 이름.
    부작용: 외부 module로 해석되면 ``ModelRuntimeError``를 발생시킨다.
    """
    module_path = Path(getattr(module, "__file__", "")).resolve()
    checkout = checkout.resolve()
    if not module_path.is_file() or checkout not in module_path.parents:
        raise ModelRuntimeError(
            f"source symbol {symbol} resolved outside pinned checkout {checkout}: "
            f"{module_path}"
        )


def ros_image_to_chw(value: Any) -> Any:
    """cv_bridge 없이 ROS ``sensor_msgs/Image``를 CHW numpy 배열로 decode한다.

    입력: ROS image 또는 이미 source-native 형식인 값.
    반환: RGB 순서의 contiguous CHW 배열 또는 변환하지 않은 원래 값.
    """
    if not all(hasattr(value, name) for name in ("height", "width", "encoding", "data")):
        return value
    encoding = str(value.encoding).lower()
    encodings = {
        "mono8": (np.uint8, 1),
        "8uc1": (np.uint8, 1),
        "rgb8": (np.uint8, 3),
        "bgr8": (np.uint8, 3),
        "8uc3": (np.uint8, 3),
        "rgba8": (np.uint8, 4),
        "bgra8": (np.uint8, 4),
        "8uc4": (np.uint8, 4),
        "32fc1": (np.float32, 1),
        "32fc2": (np.float32, 2),
        "32fc3": (np.float32, 3),
        "32fc4": (np.float32, 4),
    }
    if encoding not in encodings:
        raise ModelInputError(f"unsupported ROS image encoding: {value.encoding}")
    dtype, channels = encodings[encoding]
    dtype = np.dtype(dtype).newbyteorder(">" if getattr(value, "is_bigendian", 0) else "<")
    raw = np.frombuffer(bytes(value.data), dtype=dtype)
    row_items = int(value.step) // dtype.itemsize
    expected_row_items = int(value.width) * channels
    if row_items < expected_row_items or raw.size < int(value.height) * row_items:
        raise ModelInputError(
            f"ROS image {value.encoding} has invalid step/data length for "
            f"{value.width}x{value.height}"
        )
    rows = raw[: int(value.height) * row_items].reshape(int(value.height), row_items)
    image = rows[:, :expected_row_items]
    if channels == 1:
        image = image.reshape(int(value.height), int(value.width))[None, ...]
    else:
        image = image.reshape(int(value.height), int(value.width), channels)
        if encoding.startswith("bgr"):
            image = image[..., [2, 1, 0] + list(range(3, channels))]
        image = image.transpose(2, 0, 1)
    return np.ascontiguousarray(image.astype(dtype.newbyteorder("="), copy=False))


def model_target_xy(
    value: Any, profile: dict[str, Any], *, carla_y_right: bool
) -> Any:
    """Lidar3D-4 ``PointStamped``를 source ego/base 좌표로 변환한다.

    입력: target point, static transform profile, CARLA y-right 여부.
    반환: model 입력용 ``[x, y]`` 또는 point가 아니면 원래 값.
    """
    if not hasattr(value, "point"):
        return value
    try:
        declaration = profile["static_transforms"]["T_base_from_lidar"]
        expected_frame = str(declaration["child_frame_id"])
        observed_frame = str(value.header.frame_id)
        translation = np.asarray(declaration["translation_m"], dtype=np.float64)
        quaternion = tuple(float(item) for item in declaration["quaternion_xyzw"])
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ModelInputError(
            "model target conversion requires static_transforms.T_base_from_lidar"
        ) from error
    if observed_frame != expected_frame:
        raise ModelInputError(
            f"model target expected {expected_frame} frame, got {observed_frame}"
        )
    if translation.shape != (3,) or len(quaternion) != 4:
        raise ModelInputError("T_base_from_lidar must contain translation[3] and quaternion[4]")
    rotation = _quaternion_rotation_matrix(quaternion)
    point_lidar = np.asarray(
        [float(value.point.x), float(value.point.y), float(value.point.z)],
        dtype=np.float64,
    )
    point_base = rotation @ point_lidar + translation
    y = -point_base[1] if carla_y_right else point_base[1]
    return [float(point_base[0]), float(y)]


def _quaternion_rotation_matrix(quaternion_xyzw: tuple[float, ...]) -> np.ndarray:
    """``xyzw`` quaternion을 base-from-LiDAR 회전 행렬로 변환한다.

    입력: 길이 4의 0이 아닌 quaternion.
    반환: 3×3 float64 rotation matrix.
    """
    x, y, z, w = quaternion_xyzw
    norm = x * x + y * y + z * z + w * w
    if norm <= np.finfo(float).eps:
        raise ModelInputError("T_base_from_lidar quaternion must be nonzero")
    scale = 2.0 / norm
    return np.asarray(
        [
            [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_source_state_dict(
    network: Any,
    raw_state: dict[str, Any],
    *,
    model_id: str,
    checkpoint_path: str | Path,
    documented_prefixes: tuple[str, ...],
) -> None:
    """문서화된 checkpoint prefix를 정규화하고 정확한 network match를 강제한다.

    입력: network, 원본 state dict, model id, checkpoint 경로, 허용 prefix.
    부작용: strict-load하거나 key mismatch 시 ``SourceRuntimeError``를 발생시킨다.
    """
    normalized: dict[str, Any] = {}
    for raw_key, value in raw_state.items():
        key = str(raw_key)
        for prefix in documented_prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key in normalized:
            raise SourceRuntimeError(
                f"model {model_id} checkpoint {checkpoint_path} has duplicate normalized key {key}"
            )
        normalized[key] = value

    expected = set(network.state_dict())
    observed = set(normalized)
    matched = expected & observed
    if not matched:
        raise SourceRuntimeError(
            f"model {model_id} checkpoint {checkpoint_path} has zero critical-key match; "
            f"missing={sorted(expected)} unexpected={sorted(observed)}"
        )
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise SourceRuntimeError(
            f"model {model_id} checkpoint {checkpoint_path} incompatible: "
            f"missing={missing} unexpected={unexpected}"
        )
    try:
        incompatibility = network.load_state_dict(normalized, strict=True)
    except Exception as error:
        raise SourceRuntimeError(
            f"model {model_id} checkpoint {checkpoint_path} strict load failed: {error}"
        ) from error
    missing_after = list(getattr(incompatibility, "missing_keys", ()))
    unexpected_after = list(getattr(incompatibility, "unexpected_keys", ()))
    if missing_after or unexpected_after:
        raise SourceRuntimeError(
            f"model {model_id} checkpoint {checkpoint_path} strict load incompatible: "
            f"missing={missing_after} unexpected={unexpected_after}"
        )
