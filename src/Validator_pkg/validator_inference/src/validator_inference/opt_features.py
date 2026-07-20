"""Deterministic non-perception inputs for the supported driving models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple


class FeatureContractError(ValueError):
    """Raised when a model feature cannot satisfy its declared contract."""


@dataclass(frozen=True)
class CameraCalibrationData:
    name: str
    frame_id: str
    image_width: int
    image_height: int
    k_original: Tuple[float, ...]
    distortion_model: str
    distortion_coefficients: Tuple[float, ...]
    t_lidar_from_camera: Tuple[float, ...]


@dataclass(frozen=True)
class DrivingFeatureData:
    model_id: str
    status_feature: Tuple[float, ...]
    tfv6_radar: Tuple[float, ...]
    tfv6_radar_rows: int
    tfv6_radar_cols: int
    camera_calibrations: Tuple[CameraCalibrationData, ...]
    source: str = "fallback"


@dataclass(frozen=True)
class FallbackTarget:
    topic: str
    x: float
    y: float = 0.0
    z: float = 0.0
    frame_id: str = "Lidar3D-4"


_SUPPORTED_MODELS = {
    "tfv6",
    "interfuser",
    "diffusiondrive_v2",
    "sparsedrive_v2",
}

_TARGETS = {
    "previous": FallbackTarget("/route/target_point_previous", 5.0),
    "current": FallbackTarget("/route/target_point", 15.0),
    "next": FallbackTarget("/route/target_point_next", 30.0),
}


def _require_model(model_id: str) -> None:
    if model_id not in _SUPPORTED_MODELS:
        raise FeatureContractError(f"unsupported model_id: {model_id}")


def build_empty_tf_v6_radar() -> Tuple[float, ...]:
    """Return the original TFv6 padding result for four empty radar sensors."""

    return tuple(
        value
        for sensor_id in range(4)
        for _ in range(75)
        for value in (0.0, 0.0, 0.0, 0.0, float(sensor_id))
    )


def _numeric_tuple(
    camera_name: str,
    field_name: str,
    value: Any,
    *,
    exact_length: Optional[int] = None,
) -> Tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise FeatureContractError(
            f"camera {camera_name} requires nonempty {field_name}"
        )
    if exact_length is not None and len(value) != exact_length:
        raise FeatureContractError(
            f"camera {camera_name} {field_name} must have length {exact_length}"
        )
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise FeatureContractError(
            f"camera {camera_name} {field_name} must contain numbers"
        ) from exc


def _camera_data(camera: Mapping[str, Any]) -> CameraCalibrationData:
    name = camera.get("name")
    if not isinstance(name, str) or not name:
        raise FeatureContractError("camera calibration requires a name")

    frame_id = camera.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id:
        raise FeatureContractError(f"camera {name} requires nonempty frame_id")

    image_size = camera.get("image_size_px")
    if (
        not isinstance(image_size, Sequence)
        or isinstance(image_size, (str, bytes))
        or len(image_size) != 2
    ):
        raise FeatureContractError(
            f"camera {name} image_size_px must have length 2"
        )
    try:
        width, height = (int(image_size[0]), int(image_size[1]))
    except (TypeError, ValueError) as exc:
        raise FeatureContractError(
            f"camera {name} image_size_px must contain integers"
        ) from exc
    if width <= 0 or height <= 0:
        raise FeatureContractError(
            f"camera {name} image_size_px must contain positive integers"
        )

    distortion_model = camera.get("distortion_model")
    if not isinstance(distortion_model, str) or not distortion_model:
        raise FeatureContractError(
            f"camera {name} requires nonempty distortion_model"
        )

    return CameraCalibrationData(
        name=name,
        frame_id=frame_id,
        image_width=width,
        image_height=height,
        k_original=_numeric_tuple(
            name, "K_original_3x3", camera.get("K_original_3x3"), exact_length=9
        ),
        distortion_model=distortion_model,
        distortion_coefficients=_numeric_tuple(
            name,
            "distortion_coefficients",
            camera.get("distortion_coefficients"),
        ),
        t_lidar_from_camera=_numeric_tuple(
            name,
            "T_lidar_from_camera_4x4",
            camera.get("T_lidar_from_camera_4x4"),
            exact_length=16,
        ),
    )


def _sparse_calibrations(
    profile_data: Optional[Mapping[str, Any]],
) -> Tuple[CameraCalibrationData, ...]:
    try:
        cameras = profile_data["static_runtime_inputs"]["camera_calibration"][  # type: ignore[index]
            "cameras"
        ]
    except (KeyError, TypeError) as exc:
        raise FeatureContractError(
            "sparsedrive_v2 requires inline camera calibration"
        ) from exc
    if not isinstance(cameras, list):
        raise FeatureContractError("camera calibration cameras must be a list")

    converted = []
    for camera in cameras:
        if not isinstance(camera, Mapping):
            raise FeatureContractError("each camera calibration must be an object")
        converted.append(_camera_data(camera))

    by_name = {camera.name: camera for camera in converted}
    expected_names = {"front", "left", "right"}
    if len(converted) != 3 or set(by_name) != expected_names:
        raise FeatureContractError(
            "camera calibrations require exactly front, left, and right"
        )
    return tuple(by_name[name] for name in ("front", "left", "right"))


def build_driving_feature_data(
    model_id: str,
    *,
    vx: float,
    vy: float = 0.0,
    ax: float = 0.0,
    ay: float = 0.0,
    profile_data: Optional[Mapping[str, Any]] = None,
) -> DrivingFeatureData:
    """Build model-native status and optional static/runtime compatibility data."""

    _require_model(model_id)

    if model_id in {"tfv6", "interfuser"}:
        status = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, float(vx))
    else:
        status = (
            0.0,
            1.0,
            0.0,
            0.0,
            float(vx),
            float(vy),
            float(ax),
            float(ay),
        )

    radar = build_empty_tf_v6_radar() if model_id == "tfv6" else ()
    calibrations = (
        _sparse_calibrations(profile_data)
        if model_id == "sparsedrive_v2"
        else ()
    )
    return DrivingFeatureData(
        model_id=model_id,
        status_feature=status,
        tfv6_radar=radar,
        tfv6_radar_rows=300 if radar else 0,
        tfv6_radar_cols=5 if radar else 0,
        camera_calibrations=calibrations,
    )


def build_fallback_targets(model_id: str) -> Tuple[FallbackTarget, ...]:
    """Return only the route topics consumed by the selected model."""

    _require_model(model_id)
    if model_id == "tfv6":
        return (_TARGETS["previous"], _TARGETS["current"], _TARGETS["next"])
    if model_id == "interfuser":
        return (_TARGETS["current"],)
    return ()
