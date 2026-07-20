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
    model_id: str
    data: dict[str, Any]
    profile_sha256: str
    calibration_profile_sha256: str


def _sha256_canonical_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_nonempty(camera: dict[str, Any], camera_name: str, field: str) -> Any:
    value = camera.get(field)
    if value is None or value == "" or value == []:
        raise ProfileError(f"camera {camera_name} requires nonempty {field}")
    return value


def _validate_sparse_calibration(calibration: dict[str, Any]) -> None:
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
