"""
지원 모델의 non-perception driving feature와 fallback target을 결정적으로 만든다.
입력: ego velocity/acceleration, 선택된 profile calibration.
출력(pub): 없음. DrivingFeatureData와 fallback target 값.
구독(sub): 없음.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple


class FeatureContractError(ValueError):
    """model feature가 선언된 계약을 만족하지 못할 때 발생한다."""


@dataclass(frozen=True)
class CameraCalibrationData:
    """한 SparseDrive camera의 원본 픽셀 기준 calibration을 보관한다.

    입력: frame, image size, intrinsic, distortion, camera-to-LiDAR transform.
    반환: OptFeeder가 ``CameraCalibration`` message로 직렬화할 불변 값 객체.
    """

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
    """모델 forward에 필요한 ego status와 정적 compatibility 입력을 보관한다.

    입력: model id, status vector, TFv6 radar placeholder, camera calibrations.
    반환: OptFeeder가 ``DrivingFeature``로 발행할 불변 값 객체.
    """

    model_id: str
    status_feature: Tuple[float, ...]
    tfv6_radar: Tuple[float, ...]
    tfv6_radar_rows: int
    tfv6_radar_cols: int
    camera_calibrations: Tuple[CameraCalibrationData, ...]
    source: str = "fallback"


@dataclass(frozen=True)
class FallbackTarget:
    """navigation node 부재 시 발행하는 LiDAR frame의 전방 target을 나타낸다.

    입력: topic과 ``Lidar3D-4`` 기준 좌표.
    반환: ``PointStamped``로 직렬화할 불변 target 값.
    """

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
    """요청 model id가 OptFeeder가 지원하는 집합에 속하는지 확인한다.

    입력: profile에서 선택된 model id.
    부작용: 지원하지 않으면 ``FeatureContractError``를 발생시킨다.
    """
    if model_id not in _SUPPORTED_MODELS:
        raise FeatureContractError(f"unsupported model_id: {model_id}")


def _numeric_tuple(
    camera_name: str,
    field_name: str,
    value: Any,
    *,
    exact_length: Optional[int] = None,
) -> Tuple[float, ...]:
    """calibration sequence를 길이까지 검증한 float tuple로 바꾼다.

    입력: camera 이름, field 이름, 수치 sequence, 선택적 정확한 길이.
    반환: 모델 adapter가 사용할 불변 float tuple.
    """
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
    """profile camera JSON 한 항목을 검증된 calibration 값 객체로 변환한다.

    입력: 한 camera calibration mapping.
    반환: 이미지 크기와 projection 변환을 갖춘 ``CameraCalibrationData``.
    """
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
    """inline profile에서 SparseDrive의 front·left·right calibration을 읽는다.

    입력: 선택 모델 profile data.
    반환: 고정 순서 ``front, left, right``의 calibration tuple.
    """
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
    command_index: int = 1,
    source: str = "fallback",
    profile_data: Optional[Mapping[str, Any]] = None,
) -> DrivingFeatureData:
    """ego motion을 모델별 status feature와 정적 compatibility data로 만든다.

    입력: model id, ego 속도·가속도, 선택적 profile data.
    반환: 해당 checkpoint의 status 순서와 calibration을 갖춘 driving feature.
    """

    _require_model(model_id)

    if model_id in {"tfv6", "interfuser"}:
        status = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, float(vx))
    else:
        if type(command_index) is not int or not 0 <= command_index < 4:
            raise FeatureContractError("command_index must be an integer in [0, 3]")
        command = tuple(float(index == command_index) for index in range(4))
        status = (
            *command,
            float(vx),
            float(vy),
            float(ax),
            float(ay),
        )

    calibrations = (
        _sparse_calibrations(profile_data)
        if model_id == "sparsedrive_v2"
        else ()
    )
    return DrivingFeatureData(
        model_id=model_id,
        status_feature=status,
        tfv6_radar=(),
        tfv6_radar_rows=0,
        tfv6_radar_cols=0,
        camera_calibrations=calibrations,
        source=source,
    )


def build_fallback_targets(model_id: str) -> Tuple[FallbackTarget, ...]:
    """선택 모델이 소비하는 route topic의 결정적 전방 fallback을 반환한다.

    입력: 선택된 model id.
    반환: TFv6는 이전·현재·다음, InterFuser는 현재 target tuple.
    """

    _require_model(model_id)
    if model_id == "tfv6":
        return (_TARGETS["previous"], _TARGETS["current"], _TARGETS["next"])
    if model_id == "interfuser":
        return (_TARGETS["current"],)
    return ()
