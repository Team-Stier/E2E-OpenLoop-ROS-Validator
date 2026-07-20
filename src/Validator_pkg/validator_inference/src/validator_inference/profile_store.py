"""
로컬 profile.json에서 선택 모델을 읽고 runtime 계약을 검증한다.
입력: profile 경로와 model_id.
출력(pub): 없음. SelectedProfile 또는 ProfileError.
구독(sub): 없음.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProfileError(ValueError):
    """로컬 inference profile이 지원되지 않거나 불완전할 때 발생한다."""


@dataclass(frozen=True)
class SelectedProfile:
    """선택된 모델의 독립 runtime profile과 재현성 hash를 보관한다.

    입력: model id, 복사된 profile data, 전체 profile·calibration hash.
    반환: profile loader가 engine과 OptFeeder에 전달하는 불변 값 객체.
    """

    model_id: str
    data: dict[str, Any]
    profile_sha256: str
    calibration_profile_sha256: str


def _sha256_canonical_json(value: Any) -> str:
    """JSON 값을 키 순서에 영향받지 않는 SHA-256으로 식별한다.

    입력: JSON으로 직렬화 가능한 값.
    반환: 16진수 SHA-256 digest.
    """
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_nonempty(camera: dict[str, Any], camera_name: str, field: str) -> Any:
    """camera calibration의 필수 비어 있지 않은 field를 읽는다.

    입력: camera object, camera 이름, 필수 field 이름.
    반환: 검증을 통과한 field 값.
    """
    value = camera.get(field)
    if value is None or value == "" or value == []:
        raise ProfileError(f"camera {camera_name} requires nonempty {field}")
    return value


def _validate_sparse_calibration(calibration: dict[str, Any]) -> None:
    """SparseDrive V2의 3-camera inline calibration 계약을 검증한다.

    입력: profile의 ``camera_calibration`` object.
    부작용: 누락·형상 오류가 있으면 ``ProfileError``를 발생시킨다.
    """
    cameras = calibration.get("cameras")
    if not isinstance(cameras, list):
        raise ProfileError("camera_calibration requires cameras")

    expected_names = {"front", "left", "right"}
    actual_names = {
        camera.get("name") for camera in cameras if isinstance(camera, dict)
    }
    if len(cameras) != 3 or actual_names != expected_names:
        raise ProfileError("cameras requires exactly front, left, and right")

    for camera in cameras:
        camera_name = camera["name"]
        intrinsic = _require_nonempty(camera, camera_name, "K_original_3x3")
        if not isinstance(intrinsic, list) or len(intrinsic) != 9:
            raise ProfileError(f"camera {camera_name} K_original_3x3 must have length 9")

        _require_nonempty(camera, camera_name, "distortion_model")
        _require_nonempty(camera, camera_name, "distortion_coefficients")

        transform = _require_nonempty(
            camera, camera_name, "T_lidar_from_camera_4x4"
        )
        if not isinstance(transform, list) or len(transform) != 16:
            raise ProfileError(
                f"camera {camera_name} T_lidar_from_camera_4x4 must have length 16"
            )


def load_selected_profile(
    profile_path: str | Path, model_id: str
) -> SelectedProfile:
    """로컬 profile에서 한 모델을 복사하고 실행에 필요한 계약을 검증한다.

    입력: ``profile.json`` 경로와 선택할 model id.
    반환: 공통 static transform과 재현성 hash가 포함된 ``SelectedProfile``.
    """
    payload = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    profiles = {item["id"]: item for item in payload["profiles"]}
    if model_id not in profiles:
        raise ProfileError(f"unsupported model_id: {model_id}")

    selected = copy.deepcopy(profiles[model_id])
    selected["static_transforms"] = copy.deepcopy(
        payload["common_inference_policy"]["static_transforms"]
    )
    calibration_hash = ""
    if model_id == "sparsedrive_v2":
        calibration = selected["static_runtime_inputs"]["camera_calibration"]
        _validate_sparse_calibration(calibration)
        calibration_hash = _sha256_canonical_json(calibration)

    return SelectedProfile(
        model_id,
        selected,
        _sha256_canonical_json(payload),
        calibration_hash,
    )
