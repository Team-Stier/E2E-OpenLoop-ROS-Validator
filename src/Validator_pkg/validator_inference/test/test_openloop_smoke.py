"""Launch and lightweight end-to-end contracts for open-loop inference."""

import csv
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import time

import pytest
import rospy
from sensor_msgs.msg import Image
from validator_msgs.msg import DrivingFeature

from validator_inference.csv_recorder import CsvRunWriter
from validator_inference.engine import InferenceEngine
from validator_inference.input_buffer import InputBuffer, InputSpec
from validator_inference.models.contracts import ModelPrediction
from validator_inference.path_builder import PathMessageBuilder
from validator_inference.profile_store import ProfileError
from validator_inference.worker import InferenceWorker


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[2]
PROFILE_FILE = PACKAGE_ROOT.parent / "config" / "profile.json"


def _load_script(name):
    path = PACKAGE_ROOT / "scripts" / name
    spec = spec_from_file_location(name.replace(".py", ""), path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_uses_sim_time_arguments_and_exactly_three_runtime_nodes():
    launch_path = PACKAGE_ROOT / "launch" / "openloop_inference.launch"
    launch = launch_path.read_text(encoding="utf-8")

    assert '<param name="/use_sim_time" value="true"/>' in launch
    assert '<arg name="model_id"' in launch
    assert (
        '<arg name="profile_file" '
        'default="$(find validator_inference)/../config/profile.json"/>' in launch
    )
    assert (
        '<arg name="record_dir" '
        'default="$(find validator_inference)/../../../records/openloop"/>'
        in launch
    )
    assert launch.count("<node ") == 3
    assert launch.count('type="opt_feeder_node.py"') == 1
    assert launch.count('type="inference_engine_node.py"') == 1
    assert launch.count('type="recorder_node.py"') == 1
    assert launch.count('<param name="profile_file" value="$(arg profile_file)"/>') == 2
    assert '<param name="record_dir" value="$(arg record_dir)"/>' in launch
    assert "/inference_time" not in launch

    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "install(DIRECTORY launch/" in cmake
    assert "DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch" in cmake


class _Publisher:
    def __init__(self, callback=None):
        self.messages = []
        self._callback = callback

    def publish(self, message):
        self.messages.append(message)
        if self._callback is not None:
            self._callback(message)


class _FakeConcreteModel:
    model_id = "diffusiondrive_v2"

    def forward(self, frame, *, rng_seed):
        assert frame.inputs["camera_main"].header.stamp <= frame.reference_stamp
        assert frame.inputs["driving_feature"].header.stamp == frame.reference_stamp
        assert rng_seed > 0
        return ModelPrediction([[3.0, 0.5], [6.0, 1.0]], "fake.source")


class _SelectedProfile:
    model_id = "diffusiondrive_v2"
    profile_sha256 = "1" * 64
    calibration_profile_sha256 = ""
    data = {
        "required_inputs": [
            {"key": "camera_main"},
            {"key": "driving_feature"},
        ],
        "output": {"coordinate_frame": "NAVSIM ego-relative SE2"},
    }


def test_fake_model_smoke_writes_completed_csv_and_lidar_path(tmp_path):
    stamp = rospy.Time(17, 42)
    input_buffer = InputBuffer(
        [
            InputSpec("camera_main", "/image/fake", "sensor_msgs/Image", 2),
            InputSpec(
                "driving_feature",
                "/driving_feature",
                "validator_msgs/DrivingFeature",
                2,
            ),
        ]
    )
    image = Image()
    image.header.stamp = rospy.Time(17, 0)
    feature = DrivingFeature()
    feature.header.stamp = stamp
    feature.model_id = "diffusiondrive_v2"
    input_buffer.store("camera_main", image)
    input_buffer.store("driving_feature", feature)

    writer = CsvRunWriter(tmp_path, "diffusiondrive_v2")
    metric_publisher = _Publisher(writer.append_metric)
    path_publisher = _Publisher()
    engine = InferenceEngine(
        selected_profile=_SelectedProfile(),
        input_buffer=input_buffer,
        worker=InferenceWorker(_FakeConcreteModel(), "smoke-run"),
        path_builder=PathMessageBuilder(
            {
                "parent_frame_id": "base_link",
                "child_frame_id": "Lidar3D-4",
                "translation_m": [2.0, 0.0, 1.5],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        ),
        path_publisher=path_publisher,
        metric_publisher=metric_publisher,
        run_id="smoke-run",
    )

    engine.on_timer_tick(None)
    deadline = time.monotonic() + 1.0
    while not path_publisher.messages and time.monotonic() < deadline:
        time.sleep(0.005)
        engine.on_timer_tick(None)

    writer.close()
    with writer.path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert rows
    assert any(row["outcome"] == "inference_completed" for row in rows)
    assert path_publisher.messages
    for path in path_publisher.messages:
        assert path.header.frame_id == "Lidar3D-4"
        assert path.poses
        assert all(pose.header.frame_id == "Lidar3D-4" for pose in path.poses)


def test_shipped_sparse_null_calibration_fails_before_worker_construction(monkeypatch):
    node = _load_script("inference_engine_node.py")
    worker_calls = []
    parameters = {
        "/use_sim_time": True,
        "~model_id": "sparsedrive_v2",
        "~profile_file": str(PROFILE_FILE),
        "~run_id": "sparse-smoke",
    }

    monkeypatch.setattr(
        node.rospy,
        "get_param",
        lambda name, default=None: parameters.get(name, default),
    )
    monkeypatch.setattr(node, "InferenceWorker", lambda *args: worker_calls.append(args))

    with pytest.raises(ProfileError, match="K_original_3x3"):
        node.InferenceEngineNode()

    assert worker_calls == []


def test_operator_docs_name_the_reproducible_rosbag_clock_sequence():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    design = (PROJECT_ROOT / "docs" / "inference_engine_design.md").read_text(
        encoding="utf-8"
    )
    contracts = (PROJECT_ROOT / "docs" / "model_io_contracts.md").read_text(
        encoding="utf-8"
    )
    required = (
        "catkin_make",
        "source devel/setup.bash",
        "fetch_model_assets.py --model-id",
        "roslaunch validator_inference openloop_inference.launch",
        "rosbag play --clock rosbag/2026-07-16-17-13-14.bag",
    )
    for document in (readme, design, contracts):
        assert all(command in document for command in required)
