"""Regression tests for source-native runtime input integrity."""

from dataclasses import dataclass
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from validator_inference import assets
from validator_inference import input_buffer
from validator_inference.models import contracts
from validator_inference.models.sparsedrive_v2 import _resize_crop_matrix
from validator_inference.models import tfv6


DIFFUSION_DERIVATIVE_RELATIVE_PATH = assets.DIFFUSION_DERIVATIVE_RELATIVE_PATH
DIFFUSION_SOURCE_COMMIT = getattr(assets, "DIFFUSION_SOURCE_COMMIT", None)
create_diffusiondrive_derivative = assets.create_diffusiondrive_derivative
default_project_root = assets.default_project_root
AlignedPointCloudSweep = getattr(input_buffer, "AlignedPointCloudSweep", None)
InferenceFrame = input_buffer.InferenceFrame
InputBuffer = input_buffer.InputBuffer
InputSpec = input_buffer.InputSpec
NotReady = input_buffer.NotReady
PoseSample = input_buffer.PoseSample
TfPoseHistory = input_buffer.TfPoseHistory
ModelInputError = contracts.ModelInputError
SourceRuntimeError = getattr(contracts, "SourceRuntimeError", RuntimeError)
load_source_state_dict = getattr(contracts, "load_source_state_dict", None)
model_target_xy = getattr(contracts, "model_target_xy", None)
prepare_tfv6_temporal_lidar = getattr(tfv6, "prepare_tfv6_temporal_lidar", None)


@dataclass(frozen=True)
class Stamp:
    secs: int
    nsecs: int = 0

    def to_nsec(self):
        return self.secs * 1_000_000_000 + self.nsecs


@dataclass(frozen=True)
class Header:
    stamp: Stamp
    frame_id: str = "Lidar3D-4"


@dataclass(frozen=True)
class PointCloud:
    index: int
    header: Header


def _pose(second, xyz=(0.0, 0.0, 0.0), quaternion=(0.0, 0.0, 0.0, 1.0)):
    return PoseSample(second * 1_000_000_000, xyz, quaternion)


def _matrix(pose):
    x, y, z, w = pose.quaternion_xyzw
    norm = x * x + y * y + z * z + w * w
    scale = 2.0 / norm
    matrix = np.asarray(
        [
            [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w), 0],
            [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w), 0],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    matrix[:3, 3] = pose.translation_xyz
    return matrix


def test_input_buffer_pairs_ten_sweeps_with_nonidentity_anchor_transforms():
    assert AlignedPointCloudSweep is not None
    poses = TfPoseHistory(max_bracket_gap_ms=120)
    quarter_turn = (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))
    for second in range(10):
        poses.add(
            _pose(
                second,
                xyz=(float(second), 0.25 * second, 0.0),
                quaternion=quarter_turn if second == 9 else (0.0, 0.0, 0.0, 1.0),
            )
        )
    buffer = InputBuffer(
        [InputSpec("lidar_raw", "/lidar3D", "sensor_msgs/PointCloud2", 10, 10)],
        pose_histories={"lidar_raw": poses},
    )
    for second in reversed(range(10)):
        buffer.store("lidar_raw", PointCloud(second, Header(Stamp(second))))

    frame = buffer.snapshot_at(Stamp(9), ["lidar_raw"])

    assert isinstance(frame, InferenceFrame)
    history = frame.inputs["lidar_raw_history"]
    assert all(isinstance(item, AlignedPointCloudSweep) for item in history)
    assert [item.point_cloud.index for item in history] == list(range(10))
    expected_first = np.linalg.inv(_matrix(poses.interpolate(Stamp(9)))) @ _matrix(
        poses.interpolate(Stamp(0))
    )
    np.testing.assert_allclose(
        history[0].T_anchor_lidar_from_sweep_lidar, expected_first, atol=1e-7
    )
    assert not np.allclose(history[0].T_anchor_lidar_from_sweep_lidar, np.eye(4))
    np.testing.assert_allclose(
        history[-1].T_anchor_lidar_from_sweep_lidar, np.eye(4), atol=1e-7
    )


def test_input_buffer_reports_missing_paired_pose_as_not_ready():
    poses = TfPoseHistory(max_bracket_gap_ms=120)
    for second in range(10):
        if second != 5:
            poses.add(_pose(second))
    buffer = InputBuffer(
        [InputSpec("lidar_raw", "/lidar3D", "sensor_msgs/PointCloud2", 10, 10)],
        pose_histories={"lidar_raw": poses},
    )
    for second in range(10):
        buffer.store("lidar_raw", PointCloud(second, Header(Stamp(second))))

    result = buffer.snapshot_at(Stamp(9), ["lidar_raw"])

    assert isinstance(result, NotReady)
    assert result.missing_input_keys == ["lidar_raw pose@5000000000"]


def test_tfv6_pointcloud_history_without_pose_source_is_not_ready():
    buffer = InputBuffer(
        [InputSpec("lidar_raw", "/lidar3D", "sensor_msgs/PointCloud2", 10, 10)]
    )
    for second in range(10):
        buffer.store("lidar_raw", PointCloud(second, Header(Stamp(second))))

    result = buffer.snapshot_at(Stamp(9), ["lidar_raw"])

    assert isinstance(result, NotReady)
    assert result.missing_input_keys == ["lidar_raw pose_history"]


def test_tfv6_temporal_preparation_reads_xyz_aligns_quantizes_and_roundtrips():
    assert AlignedPointCloudSweep is not None
    assert prepare_tfv6_temporal_lidar is not None
    history = [
        AlignedPointCloudSweep(
            PointCloud(index, Header(Stamp(index))),
            np.asarray(
                [[1, 0, 0, float(index)], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float64,
            ),
        )
        for index in range(10)
    ]
    events = []

    def read_points(message, field_names, skip_nans):
        events.append(("read", message.index, field_names, skip_nans))
        return iter([(0.14, 0.26, 0.34), (float("nan"), 2.0, 3.0)])

    def rasterize(config, lidar):
        events.append(("raster", lidar.copy()))
        return np.asarray([[0.25, 0.5]], dtype=np.float32)

    def compress(image, config):
        events.append(("compress", image.copy()))
        return b"source-png"

    def decompress(payload):
        events.append(("decompress", payload))
        return np.asarray([[0.2500038, 0.5000076]], dtype=np.float32)

    result = prepare_tfv6_temporal_lidar(
        history,
        training_config=object(),
        point_precision_xyz=(0.1, 0.1, 0.1),
        read_points=read_points,
        rasterize_lidar=rasterize,
        compress_float_image=compress,
        decompress_float_image=decompress,
    )

    assert [event[0] for event in events] == ["read"] * 10 + [
        "raster",
        "compress",
        "decompress",
    ]
    raster_input = events[10][1]
    assert raster_input.shape == (10, 3)
    np.testing.assert_allclose(raster_input[0], [0.1, -0.3, 0.3])
    np.testing.assert_allclose(raster_input[-1], [9.1, -0.3, 0.3])
    np.testing.assert_allclose(result, [0.2500038, 0.5000076])


def _point(x, y, frame="Lidar3D-4"):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame),
        point=SimpleNamespace(x=x, y=y, z=0.0),
    )


def _target_profile(translation, quaternion=(0.0, 0.0, 0.0, 1.0)):
    return {
        "static_transforms": {
            "T_base_from_lidar": {
                "parent_frame_id": "base_link",
                "child_frame_id": "Lidar3D-4",
                "translation_m": list(translation),
                "quaternion_xyzw": list(quaternion),
            }
        }
    }


@pytest.mark.parametrize(("distance", "expected_x"), [(5.0, 7.0), (15.0, 17.0), (30.0, 32.0)])
def test_carla_target_conversion_applies_observed_lidar_mount(distance, expected_x):
    assert model_target_xy is not None
    result = model_target_xy(
        _point(distance, 0.0),
        _target_profile((2.0, 0.0, 1.5)),
        carla_y_right=True,
    )
    assert result == pytest.approx([expected_x, -0.0])


def test_target_conversion_applies_rotation_and_rejects_unexpected_frame():
    assert model_target_xy is not None
    quarter_turn = (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))
    profile = _target_profile((2.0, 3.0, 1.5), quarter_turn)
    assert model_target_xy(_point(5.0, 0.0), profile, carla_y_right=True) == pytest.approx(
        [2.0, -8.0]
    )
    assert model_target_xy(_point(5.0, 0.0), profile, carla_y_right=False) == pytest.approx(
        [2.0, 8.0]
    )
    with pytest.raises(ModelInputError, match=r"expected Lidar3D-4.*map"):
        model_target_xy(_point(5.0, 0.0, frame="map"), profile, carla_y_right=True)


def test_sparse_resize_crop_uses_canonical_1920x1080_geometry():
    np.testing.assert_allclose(
        _resize_crop_matrix(640, 480),
        [[0.8, 0.0, 0.0], [0.0, 0.6, -32.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(
        _resize_crop_matrix(1280, 720),
        [[0.4, 0.0, 0.0], [0.0, 0.4, -32.0], [0.0, 0.0, 1.0]],
    )


class _StateNetwork:
    def __init__(self, keys):
        self._state = {key: object() for key in keys}
        self.loaded = None

    def state_dict(self):
        return self._state

    def load_state_dict(self, state, strict):
        self.loaded = (dict(state), strict)
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


def test_source_checkpoint_loader_accepts_documented_prefix_and_full_match(tmp_path):
    assert load_source_state_dict is not None
    network = _StateNetwork(["encoder.weight", "head.bias"])
    state = {
        "agent._transfuser_model.encoder.weight": 1,
        "agent._transfuser_model.head.bias": 2,
    }
    load_source_state_dict(
        network,
        state,
        model_id="diffusiondrive_v2",
        checkpoint_path=tmp_path / "model.ckpt",
        documented_prefixes=("agent._transfuser_model.",),
    )
    assert network.loaded == ({"encoder.weight": 1, "head.bias": 2}, True)


@pytest.mark.parametrize(
    ("state", "error_fragment"),
    [
        ({"wrong.weight": 1}, "missing=.*encoder.weight.*unexpected=.*wrong.weight"),
        ({"encoder.weight": 1, "head.bias": 2, "extra": 3}, "unexpected=.*extra"),
        ({}, "zero critical-key match"),
    ],
)
def test_source_checkpoint_loader_rejects_mismatch_and_zero_match(
    tmp_path, state, error_fragment
):
    assert load_source_state_dict is not None
    network = _StateNetwork(["encoder.weight", "head.bias"])
    with pytest.raises(SourceRuntimeError, match=error_fragment) as caught:
        load_source_state_dict(
            network,
            state,
            model_id="sparsedrive_v2",
            checkpoint_path=tmp_path / "bad.ckpt",
            documented_prefixes=("agent._sparsedrive_model.",),
        )
    assert "sparsedrive_v2" in str(caught.value)
    assert str(tmp_path / "bad.ckpt") in str(caught.value)


def test_diffusion_derivative_from_actual_pinned_checkout_is_pdm_free(tmp_path):
    assert DIFFUSION_SOURCE_COMMIT is not None
    source_checkout = default_project_root() / "models/sources/diffusiondrive_v2"
    if not source_checkout.is_dir():
        pytest.skip("profile-pinned DiffusionDriveV2 source has not been fetched")
    checkout = tmp_path / "diffusiondrive_v2"
    shutil.copytree(source_checkout, checkout)

    metadata = create_diffusiondrive_derivative(checkout)

    derivative = checkout / DIFFUSION_DERIVATIVE_RELATIVE_PATH
    text = derivative.read_text(encoding="utf-8")
    compile(text, str(derivative), "exec")
    assert metadata["source_commit"] == DIFFUSION_SOURCE_COMMIT
    assert "class V2TransfuserModel" in text
    assert "forward_test_rl_without_pdm" in text
    assert 'return {"trajectory": traj_to_score[:, -1]}' in text
    for banned in (
        "pdm_planner",
        "navsim.evaluate.pdm_score",
        "ProcessPoolExecutor",
        "_pdm_pool",
        "metric_cache",
        "get_pdm_score_para",
        "reward_dict",
    ):
        assert banned not in text
    selector_line = next(
        line.strip()
        for line in text.splitlines()
        if line.strip() == 'return {"trajectory": traj_to_score[:, -1]}'
    )
    namespace = {}
    exec("def selected(traj_to_score):\n    " + selector_line, namespace)
    candidates = np.arange(2 * 4 * 8 * 3).reshape(2, 4, 8, 3)
    np.testing.assert_array_equal(
        namespace["selected"](candidates)["trajectory"], candidates[:, -1]
    )
