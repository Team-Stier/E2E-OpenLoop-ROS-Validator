import json
from pathlib import Path

import pytest

from validator_inference.profile_store import ProfileError, load_selected_profile


@pytest.fixture
def profile_path() -> Path:
    return Path(__file__).parents[2] / "config" / "profile.json"


def test_selected_profile_has_stable_hashes(profile_path: Path):
    selected = load_selected_profile(profile_path, "tfv6")
    assert selected.model_id == "tfv6"
    assert len(selected.profile_sha256) == 64
    assert selected.calibration_profile_sha256 == ""
    assert selected.data["static_transforms"]["T_base_from_lidar"] == {
        "parent_frame_id": "base_link",
        "child_frame_id": "Lidar3D-4",
        "translation_m": [2.0, 0.0, 1.5],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def test_tfv6_profile_pins_official_noradar_checkpoint_contract(profile_path: Path):
    tfv6 = load_selected_profile(profile_path, "tfv6").data
    checkpoint_assets = tfv6["runtime_assets"][:3]
    carla_agent_assets = tfv6["runtime_assets"][3:]

    assert tfv6["source"]["commit"] == (
        "730bc1a2f44d5f28312dd55f0ca958e94a24c038"
    )
    assert tfv6["source"]["compatible_public_branch"] == "cvpr2026"
    assert tfv6["reference_variant"] == "ln2697/tfv6 noradar_resnet34"
    assert [asset["path"] for asset in checkpoint_assets] == [
        "models/tfv6/noradar_resnet34/config.json",
        "models/tfv6/noradar_resnet34/model_0030_1.pth",
        "models/tfv6/noradar_resnet34/model_0030_2.pth",
    ]
    assert [asset["source_path"] for asset in checkpoint_assets] == [
        "noradar_resnet34/config.json",
        "noradar_resnet34/model_0030_1.pth",
        "noradar_resnet34/model_0030_2.pth",
    ]
    assert all(asset["sha256"] is None for asset in checkpoint_assets)
    carla_runtime_root = (
        "models/sources/lead/3rd_party/CARLA_0915/PythonAPI/carla"
    )
    carla_source_root = (
        "https://raw.githubusercontent.com/carla-simulator/carla/"
        "d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18/PythonAPI/carla"
    )
    carla_files = [
        "agents/__init__.py",
        "agents/navigation/__init__.py",
        "agents/navigation/local_planner.py",
        "agents/navigation/controller.py",
        "agents/tools/__init__.py",
        "agents/tools/misc.py",
    ]
    assert [asset["path"] for asset in carla_agent_assets] == [
        f"{carla_runtime_root}/{relative}" for relative in carla_files
    ]
    assert [asset["source_url"] for asset in carla_agent_assets] == [
        f"{carla_source_root}/{relative}" for relative in carla_files
    ]
    assert [asset["sha256"] for asset in carla_agent_assets] == [
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "7a5db9112b0d107038765abfb87b2d7a418b98e4d2c0aa92a4aade53c95b319f",
        "f95f427a5ad5ba50a96f6971df72a5f3ec078c8457b0ba2a12558396e8b5c402",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "7e453c513c742a1ff3c8f3604fe55adafac2c682753093262e7f402e9ec1b72b",
    ]
    assert tfv6["asset_checksum_policy"] == "compute_and_record_after_download"
    assert tfv6["radar_input"]["enabled"] is False
    assert tfv6["radar_input"]["checkpoint_variant"] == "official_noradar_resnet34"
    assert "radar_compatibility" not in tfv6
    assert "synthetic_radar_empty" not in str(tfv6)
    assert all(item["key"] != "radar" for item in tfv6["required_inputs"])


def test_interfuser_profile_preserves_exact_asset_hashes_and_provenance(
    profile_path: Path,
):
    interfuser = load_selected_profile(profile_path, "interfuser").data

    assert interfuser["runtime_assets"] == [
        {
            "role": "checkpoint",
            "path": "models/interfuser/official_carla/interfuser.pth.tar",
            "expected_filename": "interfuser.pth.tar",
            "sha256": (
                "39548dcfa56cdc06b8ccc52d7f6187a7d70ef36b05b7027db4d64a57ac3e30b2"
            ),
            "provenance": {
                "classification": "third_party_compatibility_artifact",
                "repository": "https://github.com/MingfeiCheng/Drivora",
                "commit": "25d56d2300f422d502c6549695b9e49efd5e5e00",
                "google_drive_id": "1GKiASmGPbD4FwHkUoVfGRk_lMLrGb2f6",
            },
        },
        {
            "role": "image_backbone",
            "path": (
                "models/shared/torch-cache/hub/checkpoints/"
                "resnet50d_ra2-464e36ba.pth"
            ),
            "source_url": (
                "https://github.com/rwightman/pytorch-image-models/releases/"
                "download/v0.1-weights/resnet50d_ra2-464e36ba.pth"
            ),
            "sha256": (
                "464e36baa940a80971e717454e61e8f6fc1a1b284b7a3dbb6f6fd4c3da78533d"
            ),
        },
    ]


def test_sparse_profile_contains_morai_inline_calibration(profile_path: Path):
    sparse = load_selected_profile(profile_path, "sparsedrive_v2")
    cameras = sparse.data["static_runtime_inputs"]["camera_calibration"]["cameras"]
    by_name = {camera["name"]: camera for camera in cameras}

    assert len(sparse.calibration_profile_sha256) == 64
    assert by_name["front"]["K_original_3x3"] == [
        640.0,
        0.0,
        640.0,
        0.0,
        640.0,
        360.0,
        0.0,
        0.0,
        1.0,
    ]
    assert by_name["left"]["K_original_3x3"][0] == pytest.approx(
        149.21845060959956
    )
    assert by_name["left"]["T_lidar_from_camera_4x4"] == pytest.approx(
        [
            0.9254165783983233,
            -0.17364817766693033,
            0.33682408883346526,
            -0.85,
            -0.3420201433256688,
            0.0,
            0.9396926207859083,
            0.65,
            -0.1631759111665348,
            -0.984807753012208,
            -0.05939117461388472,
            -0.3,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    )
    assert by_name["right"]["T_lidar_from_camera_4x4"] == pytest.approx(
        [
            -0.9254165783983233,
            -0.17364817766693033,
            0.33682408883346543,
            -0.85,
            -0.342020143325669,
            0.0,
            -0.9396926207859083,
            -0.65,
            0.1631759111665348,
            -0.984807753012208,
            -0.05939117461388475,
            -0.3,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    )
    assert all(camera["distortion_model"] == "plumb_bob" for camera in cameras)
    assert all(
        camera["distortion_coefficients"] == [0.0] * 5 for camera in cameras
    )
    assert sparse.data["readiness"]["state"] == "ready_after_runtime_assets"


def test_sparse_profile_rejects_partial_inline_calibration(
    profile_path: Path, tmp_path: Path
):
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    sparse = next(item for item in payload["profiles"] if item["id"] == "sparsedrive_v2")
    sparse["static_runtime_inputs"]["camera_calibration"]["cameras"][0][
        "K_original_3x3"
    ] = None
    incomplete = tmp_path / "profile.json"
    incomplete.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProfileError, match="K_original_3x3"):
        load_selected_profile(incomplete, "sparsedrive_v2")
