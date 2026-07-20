"""원본 코드 기반 runtime 입력 무결성 회귀를 검사한다."""

import ast
import copy
from dataclasses import dataclass
import io
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from validator_inference import assets
from validator_inference import input_buffer
from validator_inference.models import contracts
from validator_inference.models import diffusiondrive_v2
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
load_diffusiondrive_selector_state_dict = getattr(
    diffusiondrive_v2, "load_diffusiondrive_selector_state_dict", None
)
selector_flow_stages = getattr(assets, "selector_flow_stages", None)


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
        expert_config=SimpleNamespace(
            ego_extent_x=0.0,
            ego_extent_y=0.0,
            save_only_non_ground_lidar=False,
            save_lidar_only_inside_bev=False,
        ),
        T_base_from_lidar={
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        point_precision_xyz=(0.1, 0.1, 0.1),
        read_points=read_points,
        remove_ground=lambda *args, **kwargs: None,
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


def test_tfv6_preaccumulation_applies_mount_ego_ransac_and_bounds_in_source_order():
    """ROS port는 accumulation 전에 LEAD base_agent.py:140-237을 유지해야 한다."""
    assert prepare_tfv6_temporal_lidar is not None
    events = []
    raw_points = [
        (0.0, 3.0, 0.0),       # inside ego after +2m mount; must stay removed
        (3.0, 3.0, -1.5),      # base z=0 ground
        (3.04, 3.04, 0.04),    # sole valid point; shows 0.1m quantization
        (100.0, 3.0, 0.0),     # x outside source BEV
        (3.0, 100.0, 0.0),     # y outside source BEV
        (3.0, 3.0, 20.0),      # z outside source height
    ]

    history = [
        AlignedPointCloudSweep(
            PointCloud(0, Header(Stamp(0))),
            np.asarray(
                [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float64,
            ),
        ),
        AlignedPointCloudSweep(
            PointCloud(1, Header(Stamp(1))), np.eye(4, dtype=np.float64)
        ),
    ]
    expert_config = SimpleNamespace(
        ego_extent_x=2.4508416652679443,
        ego_extent_y=1.0641621351242065,
        save_only_non_ground_lidar=True,
        save_lidar_only_inside_bev=True,
        min_x_meter=-32.0,
        max_x_meter=64.0,
        min_y_meter=-40.0,
        max_y_meter=40.0,
        min_height_lidar=-4.0,
        max_height_lidar=10.0,
    )

    def read_points(message, field_names, skip_nans):
        return iter(raw_points)

    def remove_ground(points, config, parallel):
        events.append(("ransac", points.copy(), config, parallel))
        return np.isclose(points[:, 2], 0.0)

    def rasterize(config, lidar):
        events.append(("raster", lidar.copy()))
        return np.asarray([[0.5]], dtype=np.float32)

    def compress(image, config):
        events.append(("compress", image.copy()))
        return b"16-bit-png"

    def decompress(payload):
        events.append(("decompress", payload))
        return np.asarray([[[0.5000076]]], dtype=np.float32)

    result = prepare_tfv6_temporal_lidar(
        history,
        training_config=object(),
        expert_config=expert_config,
        T_base_from_lidar={
            "translation_m": [2.0, 0.0, 1.5],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        point_precision_xyz=(0.1, 0.1, 0.1),
        read_points=read_points,
        remove_ground=remove_ground,
        rasterize_lidar=rasterize,
        compress_float_image=compress,
        decompress_float_image=decompress,
    )

    assert [event[0] for event in events] == [
        "ransac", "ransac", "raster", "compress", "decompress"
    ]
    # RANSAC은 원본과 동일한 ego-footprint mask 뒤, bounds/alignment 전에
    # 전체 +2,+0,+1.5 mount를 반영한 base/ego 좌표를 본다.
    assert events[0][1][0] == pytest.approx([5.0, 3.0, 0.0])
    assert all(not np.isclose(row[0], 2.0) for row in events[0][1])
    assert events[0][3] is True
    # 오래된 sweep은 filtering 뒤에만 align하고, ROS y-left는 raster 경계에서 한 번 뒤집는다.
    np.testing.assert_allclose(
        events[2][1],
        [[15.0, -3.0, 1.5], [5.0, -3.0, 1.5]],
        atol=1e-6,
    )
    assert result == pytest.approx(0.5000076)


def test_tfv6_port_is_tied_to_pinned_lead_prefilter_and_png_helpers():
    checkout = default_project_root() / "models/sources/lead"
    if not checkout.is_dir():
        pytest.skip("profile-pinned LEAD source has not been fetched")
    assert shutil.which("git")
    import subprocess

    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == "730bc1a2f44d5f28312dd55f0ca958e94a24c038"
    base_agent = (checkout / "lead/common/base_agent.py").read_text(encoding="utf-8")
    sensor_agent = (checkout / "lead/inference/sensor_agent.py").read_text(encoding="utf-8")
    cache = (checkout / "lead/data_loader/training_cache.py").read_text(encoding="utf-8")
    for source_fragment in (
        "common_utils.lidar_to_ego_coordinate(",
        "(np.abs(lidar_x) > self.config_expert.ego_extent_x)",
        "ground_mask = ransac.remove_ground(",
        "self.config_expert.save_lidar_only_inside_bev",
        "common_utils.align_lidar(",
    ):
        assert source_fragment in base_agent
    assert "np.round(lidar[:, 0] / self.config_expert.point_precision_x)" in sensor_agent
    assert "training_cache.compress_float_image(" in sensor_agent
    assert "training_cache.decompress_float_image(" in sensor_agent
    assert "(image * 65535.0).astype(np.uint16)" in cache


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


class _SchemaOnlyUnpickler(pickle.Unpickler):
    """tensor storage를 실체화하지 않고 torch checkpoint key 구조를 읽는다."""

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda *args: None
        if module == "torch" and name in {"FloatStorage", "LongStorage"}:
            return type(name, (), {})
        return super().find_class(module, name)

    def persistent_load(self, persistent_id):
        return None


def _official_ddv2_schema_or_fixture():
    checkpoint = Path(
        "/private/tmp/task5-checkpoint-schema-review/diffusiondrivev2_sel.ckpt"
    )
    if checkpoint.is_file():
        with zipfile.ZipFile(checkpoint) as archive:
            data_pickle = next(
                name for name in archive.namelist() if name.endswith("/data.pkl")
            )
            payload = _SchemaOnlyUnpickler(
                io.BytesIO(archive.read(data_pickle))
            ).load()
        state = payload["state_dict"]
        assert len(state) == 972
        return {key: object() for key in state}, checkpoint
    fixture_keys = [
        f"agent._transfuser_model.fixture.layer_{index}.weight"
        for index in range(971)
    ]
    fixture_keys.append("agent._transfuser_model._trajectory_head.vocab")
    return {key: object() for key in fixture_keys}, Path("fixture.ckpt")


class _NumpyTensor(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def unsqueeze(self, axis):
        return _NumpyTensor(np.expand_dims(self, axis))


class _TorchSelectorSeam:
    @staticmethod
    def argmax(value, dim=-1):
        return _NumpyTensor(np.argmax(np.asarray(value), axis=dim))

    @staticmethod
    def arange(size, device=None):
        return _NumpyTensor(np.arange(size))

    @staticmethod
    def cat(values, dim=0):
        return _NumpyTensor(np.concatenate(values, axis=dim))


def _generated_selector_tail(source_text):
    """생성된 method의 learned-selector 뒷부분 전체를 CPU seam으로 컴파일한다."""
    tree = ast.parse(source_text)
    trajectory = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TrajectoryHead"
    )
    method = next(
        node
        for node in trajectory.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "forward_test_rl_without_pdm"
    )
    start = next(
        index
        for index, statement in enumerate(method.body)
        if "_get_scorer_inputs" in (ast.get_source_segment(source_text, statement) or "")
    )
    seam = ast.parse(
        """def run_selector(self, diffusion_output):
    bs = diffusion_output.shape[0]
    device = None
    bev_feature = None
    bev_spatial_shape = None
    agents_query = None
    ego_query = None
    status_encoding = None
    global_img = None
"""
    ).body[0]
    seam.body.extend(copy.deepcopy(method.body[start:]))
    module = ast.fix_missing_locations(ast.Module(body=[seam], type_ignores=[]))
    namespace = {"torch": _TorchSelectorSeam}
    exec(compile(module, "<generated-ddv2-selector-seam>", "exec"), namespace)
    return namespace["run_selector"]


class _SelectorHeadSeam:
    def __init__(self):
        self.events = []
        self.NC_head = self.EP_head = self.DAC_head = self.TTC_head = self.C_head = lambda value: value
        self.fine_NC_head = self.fine_EP_head = self.fine_DAC_head = lambda value: value
        self.fine_TTC_head = self.fine_C_head = lambda value: value
        self.sigmoid = lambda value: value

    def _get_scorer_inputs(self, diffusion_output, batch_size, group_count):
        self.events.append("diffusion_denoising")
        score = _NumpyTensor(
            np.arange(1, group_count + 1, dtype=np.float64)[None, :, None]
        )
        return score, score, None

    def scorer_decoder(self, *args):
        self.events.append("coarse_scorer")
        return [args[0]]

    def _select_topk(self, **kwargs):
        assert kwargs["topk"] == 32
        self.events.append("top_32")
        topk_idx = _NumpyTensor(np.arange(32)[None, :])
        return kwargs["traj_feature"][:, :32], kwargs["noisy_traj_points_xy"][:, :32], None, topk_idx, None

    def fine_scorer_decoder(self, *args):
        self.events.append("fine_scorer")
        score = _NumpyTensor(np.arange(1, 33, dtype=np.float64)[None, :, None])
        return [score, score, score]


def test_ddv2_loader_allows_only_removed_pdm_vocab_and_strictly_matches_971_keys():
    assert load_diffusiondrive_selector_state_dict is not None
    checkpoint_state, checkpoint_path = _official_ddv2_schema_or_fixture()
    normalized_keys = [
        key[len("agent._transfuser_model.") :]
        for key in checkpoint_state
        if key != "agent._transfuser_model._trajectory_head.vocab"
    ]
    network = _StateNetwork(normalized_keys)

    load_diffusiondrive_selector_state_dict(
        network, checkpoint_state, checkpoint_path=checkpoint_path
    )

    assert len(network.loaded[0]) == 971
    assert network.loaded[1] is True
    assert "_trajectory_head.vocab" not in network.loaded[0]


@pytest.mark.parametrize("mutation", ["extra", "missing", "wrong_prefix"])
def test_ddv2_loader_rejects_every_non_vocab_schema_mismatch(mutation):
    assert load_diffusiondrive_selector_state_dict is not None
    raw = {
        "agent._transfuser_model.encoder.weight": object(),
        "agent._transfuser_model.head.bias": object(),
        "agent._transfuser_model._trajectory_head.vocab": object(),
    }
    network = _StateNetwork(["encoder.weight", "head.bias"])
    if mutation == "extra":
        raw["agent._transfuser_model.extra"] = object()
    elif mutation == "missing":
        raw.pop("agent._transfuser_model.head.bias")
    else:
        raw["not_documented.encoder.weight"] = raw.pop(
            "agent._transfuser_model.encoder.weight"
        )

    with pytest.raises(SourceRuntimeError, match=r"diffusiondrive_v2.*schema.ckpt"):
        load_diffusiondrive_selector_state_dict(
            network, raw, checkpoint_path=Path("schema.ckpt")
        )


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
    assert selector_flow_stages is not None
    assert selector_flow_stages(text) == (
        "diffusion_denoising",
        "coarse_scorer",
        "top_32",
        "fine_scorer",
        "final_trajectory",
    )
    selector = _generated_selector_tail(text)
    seam = _SelectorHeadSeam()
    candidates = _NumpyTensor(np.arange(200 * 8 * 3).reshape(1, 200, 8, 3))

    output = selector(seam, candidates)

    assert seam.events == [
        "diffusion_denoising", "coarse_scorer", "top_32", "fine_scorer"
    ]
    np.testing.assert_array_equal(output["trajectory"], candidates[:, 31])
