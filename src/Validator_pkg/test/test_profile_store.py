from pathlib import Path

import pytest

from validator_inference.profile_store import ProfileError, load_selected_profile


@pytest.fixture
def profile_path() -> Path:
    return Path(__file__).parents[1] / "config" / "profile.json"


def test_selected_profile_has_stable_hashes(profile_path: Path):
    selected = load_selected_profile(profile_path, "tfv6")
    assert selected.model_id == "tfv6"
    assert len(selected.profile_sha256) == 64
    assert selected.calibration_profile_sha256 == ""


def test_sparse_profile_rejects_partial_inline_calibration(profile_path: Path):
    with pytest.raises(ProfileError, match="K_original_3x3"):
        load_selected_profile(profile_path, "sparsedrive_v2")
