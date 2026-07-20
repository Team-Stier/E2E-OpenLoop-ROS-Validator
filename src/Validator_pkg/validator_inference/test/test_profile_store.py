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


def test_sparse_profile_rejects_partial_inline_calibration(profile_path: Path):
    with pytest.raises(ProfileError, match="K_original_3x3"):
        load_selected_profile(profile_path, "sparsedrive_v2")
