import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from validator_inference import assets
from validator_inference.assets import (
    AssetError,
    fetch_runtime_assets,
    verify_runtime_assets,
    write_asset_manifest,
)
from validator_inference.input_buffer import InferenceFrame
from validator_inference.models.contracts import ModelInputError
from validator_inference.models.diffusiondrive_v2 import (
    DiffusionDriveV2InferenceModel,
)
from validator_inference.models.interfuser import InterFuserModel
from validator_inference.models.select import build_selected_model
from validator_inference.models.sparsedrive_v2 import SparseDriveV2Model
from validator_inference.models.tfv6 import TFv6Model
from validator_inference.profile_store import SelectedProfile, load_selected_profile


MODEL_TYPES = {
    "tfv6": TFv6Model,
    "interfuser": InterFuserModel,
    "diffusiondrive_v2": DiffusionDriveV2InferenceModel,
    "sparsedrive_v2": SparseDriveV2Model,
}
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = PACKAGE_ROOT.parent / "config" / "profile.json"


def test_generated_asset_publication_preserves_equal_file_identity(tmp_path):
    destination = tmp_path / "generated.py"
    payload = b"VALUE = 1\n"
    destination.write_bytes(payload)
    before = destination.stat()

    assets._publish_generated_bytes(destination, payload)

    after = destination.stat()
    assert destination.read_bytes() == payload
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )


def test_generated_asset_publication_atomically_replaces_changed_bytes(
    tmp_path, monkeypatch
):
    destination = tmp_path / "generated.py"
    destination.write_bytes(b"VALUE = 1\n")
    real_replace = os.replace
    replacements = []

    def record_replace(source, target):
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(assets.os, "replace", record_replace)

    assets._publish_generated_bytes(destination, b"VALUE = 2\n")

    assert destination.read_bytes() == b"VALUE = 2\n"
    assert len(replacements) == 1
    assert replacements[0][1] == destination
    assert replacements[0][0].parent == destination.parent


def _selected(model_id, data=None):
    payload = data or {
        "id": model_id,
        "source": {
            "repository": "https://example.invalid/source.git",
            "commit": "a" * 40,
        },
        "runtime_assets": [],
        "required_inputs": [],
    }
    return SelectedProfile(model_id, payload, "b" * 64, "")


def _git_repo(path):
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )
    (path / "source.txt").write_text("source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "source.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "source"], check=True
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _frame(inputs):
    return InferenceFrame(
        reference_stamp=SimpleNamespace(to_nsec=lambda: 1),
        inputs=inputs,
        input_age_ms={key: 0.0 for key in inputs},
        history_count=10 if "lidar_raw_history" in inputs else 0,
        history_span_ms=900.0 if "lidar_raw_history" in inputs else 0.0,
    )


def _driving(status, radar=(), rows=0, cols=0, calibrations=()):
    return SimpleNamespace(
        status_feature=list(status),
        tfv6_radar=list(radar),
        tfv6_radar_rows=rows,
        tfv6_radar_cols=cols,
        camera_calibrations=list(calibrations),
    )


@pytest.mark.parametrize("model_id, expected_type", MODEL_TYPES.items())
def test_factory_returns_exact_concrete_type_without_loading_assets(
    tmp_path, model_id, expected_type
):
    model = build_selected_model(_selected(model_id), project_root=tmp_path)
    assert type(model) is expected_type


def test_factory_rejects_unknown_model_id(tmp_path):
    with pytest.raises(ValueError, match="unsupported model_id: unknown"):
        build_selected_model(_selected("unknown"), project_root=tmp_path)


def test_interfuser_asset_order_keeps_checkpoint_at_runtime_index_zero():
    profile = load_selected_profile(PROFILE_PATH, "interfuser").data

    assert [asset["role"] for asset in profile["runtime_assets"]] == [
        "checkpoint",
        "image_backbone",
    ]
    assert profile["runtime_assets"][0]["path"] == (
        "models/interfuser/official_carla/interfuser.pth.tar"
    )
    assert "43.163.208.95" not in json.dumps(profile, sort_keys=True)


def test_missing_asset_names_resolved_path_and_expected_hash(tmp_path):
    expected = "0" * 64
    profile = {
        "id": "tfv6",
        "runtime_assets": [
            {"role": "weight", "path": "models/tfv6/missing.pth", "sha256": expected}
        ],
    }
    resolved = tmp_path / "models/tfv6/missing.pth"
    with pytest.raises(AssetError) as caught:
        verify_runtime_assets(profile, project_root=tmp_path, verify_source=False)
    assert str(resolved) in str(caught.value)
    assert f"expected sha256 {expected}" in str(caught.value)


def test_wrong_hash_names_path_expected_and_observed_hash(tmp_path):
    asset = tmp_path / "models/value.bin"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"actual")
    expected = "1" * 64
    with pytest.raises(AssetError) as caught:
        verify_runtime_assets(
            {
                "id": "interfuser",
                "runtime_assets": [
                    {"role": "checkpoint", "path": "models/value.bin", "sha256": expected}
                ],
            },
            project_root=tmp_path,
            verify_source=False,
        )
    assert str(asset) in str(caught.value)
    assert f"expected sha256 {expected}" in str(caught.value)
    assert hashlib.sha256(b"actual").hexdigest() in str(caught.value)


def test_null_declared_hash_is_accepted_and_observed_hash_is_recorded(tmp_path):
    asset = tmp_path / "models/value.bin"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"unconstrained")
    metadata = verify_runtime_assets(
        {
            "id": "interfuser",
            "runtime_assets": [
                {"role": "checkpoint", "path": "models/value.bin", "sha256": None}
            ],
        },
        project_root=tmp_path,
        verify_source=False,
    )
    assert metadata["assets"][0]["declared_sha256"] is None
    assert metadata["assets"][0]["observed_sha256"] == hashlib.sha256(
        b"unconstrained"
    ).hexdigest()


def test_relative_paths_resolve_from_project_root_not_cwd(tmp_path, monkeypatch):
    asset = tmp_path / "models/value.bin"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"root-relative")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    metadata = verify_runtime_assets(
        {
            "id": "tfv6",
            "runtime_assets": [
                {"role": "config", "path": "models/value.bin", "sha256": None}
            ],
        },
        project_root=tmp_path,
        verify_source=False,
    )
    assert metadata["assets"][0]["resolved_path"] == str(asset)


def test_source_checkout_revision_mismatch_is_rejected(tmp_path):
    checkout = tmp_path / "models/sources/lead"
    observed = _git_repo(checkout)
    expected = "f" * 40
    with pytest.raises(AssetError) as caught:
        verify_runtime_assets(
            {
                "id": "tfv6",
                "source": {
                    "repository": "https://example.invalid/lead",
                    "commit": expected,
                },
                "runtime_assets": [],
            },
            project_root=tmp_path,
        )
    assert "tfv6" in str(caught.value)
    assert expected in str(caught.value)
    assert observed in str(caught.value)


def test_manifest_is_canonical_and_contains_source_and_observed_hash(tmp_path):
    checkout = tmp_path / "models/sources/interfuser"
    commit = _git_repo(checkout)
    asset = tmp_path / "models/interfuser/official/model.pth"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"checkpoint")
    profile = {
        "id": "interfuser",
        "source": {"repository": "file:///upstream", "commit": commit},
        "runtime_assets": [
            {"role": "checkpoint", "path": "models/interfuser/official/model.pth", "sha256": None}
        ],
    }
    metadata = verify_runtime_assets(profile, project_root=tmp_path)
    path = write_asset_manifest(profile, metadata, project_root=tmp_path)
    first = path.read_bytes()
    write_asset_manifest(profile, metadata, project_root=tmp_path)
    assert path.read_bytes() == first
    payload = json.loads(first)
    assert payload["model_id"] == "interfuser"
    assert payload["source"]["commit"] == commit
    assert payload["assets"][0]["observed_sha256"] == hashlib.sha256(
        b"checkpoint"
    ).hexdigest()
    assert first.endswith(b"\n")


def test_fetch_dispatches_download_atomically_and_writes_manifest(tmp_path):
    upstream = tmp_path / "upstream"
    commit = _git_repo(upstream)
    profile = {
        "id": "interfuser",
        "source": {"repository": upstream.as_uri(), "commit": commit},
        "runtime_assets": [
            {
                "role": "checkpoint",
                "path": "models/interfuser/official/model.pth",
                "source_url": "file:///fixture/model.pth",
                "sha256": hashlib.sha256(b"downloaded").hexdigest(),
            }
        ],
    }
    calls = []

    def downloader(url, destination):
        calls.append((url, Path(destination)))
        Path(destination).write_bytes(b"downloaded")

    manifest = fetch_runtime_assets(
        profile,
        model_id="interfuser",
        project_root=tmp_path,
        downloader=downloader,
    )
    target = tmp_path / "models/interfuser/official/model.pth"
    assert [call[0] for call in calls] == ["file:///fixture/model.pth"]
    assert target.read_bytes() == b"downloaded"
    assert all(call[1] != target for call in calls)
    assert manifest == tmp_path / "models/interfuser/asset-manifest.json"


def test_load_is_lazy_but_missing_source_failure_names_model_and_checkout(tmp_path):
    model = build_selected_model(
        _selected(
            "tfv6",
            {
                "id": "tfv6",
                "source": {"repository": "file:///lead", "commit": "a" * 40},
                "runtime_assets": [],
                "required_inputs": [],
            },
        ),
        project_root=tmp_path,
    )
    with pytest.raises(RuntimeError) as caught:
        model.load()
    assert "tfv6" in str(caught.value)
    assert str(tmp_path / "models/sources/lead") in str(caught.value)


class _TFv6Source:
    def __init__(self):
        self.received = None

    def build_temporal_lidar(self, history):
        assert len(history) == 10
        assert all(hasattr(item, "point_cloud") for item in history)
        assert all(
            hasattr(item, "T_anchor_lidar_from_sweep_lidar") for item in history
        )
        return "source-rasterized-history"

    def forward(self, data):
        self.received = data
        return SimpleNamespace(pred_future_waypoints=np.ones((1, 8, 2)))


def test_tfv6_forward_uses_history_and_has_no_radar_model_input(tmp_path):
    source = _TFv6Source()
    model = TFv6Model(_selected("tfv6"), tmp_path, source_runtime=source)
    feature = _driving([0, 0, 0, 1, 0, 0, 7.0])
    prediction = model.forward(
        _frame(
            {
                "camera_main": "panorama",
                "lidar_raw_history": [
                    SimpleNamespace(
                        point_cloud=f"cloud-{index}",
                        T_anchor_lidar_from_sweep_lidar=np.eye(4),
                    )
                    for index in range(10)
                ],
                "driving_feature": feature,
                "route_target_point_previous": "previous",
                "route_target_point": "current",
                "route_target_point_next": "next",
            }
        )
    )
    assert prediction.trajectory.shape == (1, 8, 2)
    assert source.received["rasterized_lidar"] == "source-rasterized-history"
    assert "radar" not in source.received
    assert source.received["command"] == [0, 0, 0, 1, 0, 0]


def test_tfv6_rejects_nonempty_radar_and_short_history(tmp_path):
    model = TFv6Model(_selected("tfv6"), tmp_path, source_runtime=_TFv6Source())
    base = {
        "camera_main": "panorama",
        "lidar_raw_history": list(range(9)),
        "driving_feature": _driving([0, 0, 0, 1, 0, 0, 7.0]),
        "route_target_point_previous": "previous",
        "route_target_point": "current",
        "route_target_point_next": "next",
    }
    with pytest.raises(ModelInputError, match=r"tfv6.*lidar_raw_history.*10"):
        model.forward(_frame(base))
    base["lidar_raw_history"] = [
        SimpleNamespace(
            point_cloud=f"cloud-{index}",
            T_anchor_lidar_from_sweep_lidar=np.eye(4),
        )
        for index in range(10)
    ]
    base["driving_feature"] = _driving(
        [0, 0, 0, 1, 0, 0, 7.0], radar=[0.0], rows=1, cols=1
    )
    with pytest.raises(ModelInputError, match=r"tfv6.*radar.*empty"):
        model.forward(_frame(base))


class _InterFuserSource:
    def __init__(self):
        self.received = None

    def forward(self, data):
        self.received = data
        return ("traffic", np.ones((1, 10, 2)), "junction", "light", "stop", "bev")


def test_interfuser_calls_source_forward_with_all_named_inputs(tmp_path):
    source = _InterFuserSource()
    model = InterFuserModel(_selected("interfuser"), tmp_path, source_runtime=source)
    prediction = model.forward(
        _frame(
            {
                "camera_front_224": "front",
                "camera_left_128": "left",
                "camera_right_128": "right",
                "camera_front_center_128": "front-center-crop",
                "lidar_bev": "histogram",
                "driving_feature": _driving([0, 0, 0, 1, 0, 0, 3.0]),
                "route_target_point": "target",
            }
        )
    )
    assert prediction.trajectory.shape == (1, 10, 2)
    assert source.received == {
        "rgb": "front",
        "rgb_left": "left",
        "rgb_right": "right",
        "rgb_center": "front-center-crop",
        "lidar": "histogram",
        "measurements": [0, 0, 0, 1, 0, 0, 3.0],
        "target_point": "target",
    }


class _DiffusionSource:
    def __init__(self):
        self.received = None
        self.seeds = []
        self.pdm_called = False

    def set_seed(self, seed):
        self.seeds.append(seed)

    def forward_pdm_free(self, features):
        self.received = features
        return {"trajectory": np.ones((1, 8, 3))}


def test_diffusiondrive_uses_pdm_free_selector_and_worker_seed(tmp_path):
    source = _DiffusionSource()
    model = DiffusionDriveV2InferenceModel(
        _selected("diffusiondrive_v2"), tmp_path, source_runtime=source
    )
    prediction = model.forward(
        _frame(
            {
                "camera_main": "stitched",
                "lidar_bev": "latest-bev",
                "driving_feature": _driving([0, 1, 0, 0, 1, 2, 3, 4]),
            }
        ),
        rng_seed=1234,
    )
    assert source.seeds == [1234]
    assert source.received == {
        "camera_feature": "stitched",
        "lidar_feature": "latest-bev",
        "status_feature": [0, 1, 0, 0, 1, 2, 3, 4],
    }
    assert prediction.trajectory.shape == (1, 8, 3)
    assert source.pdm_called is False


def _calibration(name, width, height, tx=0.0):
    transform = np.eye(4)
    transform[0, 3] = tx
    return SimpleNamespace(
        name=name,
        frame_id=f"{name}-frame",
        image_width=width,
        image_height=height,
        k_original=[100.0, 0.0, width / 2, 0.0, 100.0, height / 2, 0.0, 0.0, 1.0],
        distortion_model="plumb_bob",
        distortion_coefficients=[0.1, 0.01, 0.0, 0.0, 0.0],
        t_lidar_from_camera=list(transform.reshape(-1)),
    )


class _SparseSource:
    def __init__(self):
        self.received = None

    def forward(self, features):
        self.received = features
        return ({"trajectory": np.ones((1, 8, 3))}, {})


def test_sparsedrive_builds_left_front_right_projection_conditions(tmp_path):
    source = _SparseSource()
    model = SparseDriveV2Model(
        _selected("sparsedrive_v2"), tmp_path, source_runtime=source
    )
    calibrations = [
        _calibration("front", 1280, 720, 1.0),
        _calibration("left", 640, 480, 2.0),
        _calibration("right", 640, 480, 3.0),
    ]
    prediction = model.forward(
        _frame(
            {
                "camera_left": "left-tensor",
                "camera_main": "front-tensor",
                "camera_right": "right-tensor",
                "driving_feature": _driving(
                    [0, 1, 0, 0, 1, 2, 3, 4], calibrations=calibrations
                ),
            }
        )
    )
    camera_feature = source.received["camera_feature"]
    assert camera_feature["imgs"] == [
        "left-tensor",
        "front-tensor",
        "right-tensor",
    ]
    assert camera_feature["camera_names"] == ["left", "front", "right"]
    assert np.asarray(camera_feature["projection_mat"]).shape == (3, 3, 4)
    assert camera_feature["distortion_models"] == ["plumb_bob"] * 3
    left_calibration = calibrations[1]
    intrinsic = np.asarray(left_calibration.k_original).reshape(3, 3)
    t_camera_from_lidar = np.linalg.inv(
        np.asarray(left_calibration.t_lidar_from_camera).reshape(4, 4)
    )
    source_resize_crop = np.asarray(
        [[0.8, 0.0, 0.0], [0.0, 0.6, -32.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(
        camera_feature["projection_mat"][0],
        source_resize_crop @ intrinsic @ t_camera_from_lidar[:3, :],
    )
    assert prediction.trajectory.shape == (1, 8, 3)


def test_each_concrete_forward_names_a_missing_input(tmp_path):
    cases = [
        (TFv6Model(_selected("tfv6"), tmp_path, _TFv6Source()), "camera_main"),
        (
            InterFuserModel(
                _selected("interfuser"), tmp_path, _InterFuserSource()
            ),
            "camera_front_224",
        ),
        (
            DiffusionDriveV2InferenceModel(
                _selected("diffusiondrive_v2"), tmp_path, _DiffusionSource()
            ),
            "camera_main",
        ),
        (
            SparseDriveV2Model(
                _selected("sparsedrive_v2"), tmp_path, _SparseSource()
            ),
            "camera_left",
        ),
    ]
    for model, missing_key in cases:
        with pytest.raises(ModelInputError, match=missing_key):
            if isinstance(model, DiffusionDriveV2InferenceModel):
                model.forward(_frame({}), rng_seed=1)
            else:
                model.forward(_frame({}))
