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

    assert tfv6["source"]["commit"] == (
        "730bc1a2f44d5f28312dd55f0ca958e94a24c038"
    )
    assert tfv6["source"]["compatible_public_branch"] == "cvpr2026"
    assert tfv6["reference_variant"] == "ln2697/tfv6 noradar_resnet34"
    assert [asset["path"] for asset in tfv6["runtime_assets"]] == [
        "models/tfv6/noradar_resnet34/config.json",
        "models/tfv6/noradar_resnet34/model_0030_1.pth",
        "models/tfv6/noradar_resnet34/model_0030_2.pth",
    ]
    assert [asset["source_path"] for asset in tfv6["runtime_assets"]] == [
        "noradar_resnet34/config.json",
        "noradar_resnet34/model_0030_1.pth",
        "noradar_resnet34/model_0030_2.pth",
    ]
    assert all(asset["sha256"] is None for asset in tfv6["runtime_assets"])
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


def test_sparse_profile_rejects_partial_inline_calibration(profile_path: Path):
    with pytest.raises(ProfileError, match="K_original_3x3"):
        load_selected_profile(profile_path, "sparsedrive_v2")
