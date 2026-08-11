"""10 Hz inference engine 조합의 실행 가능한 계약을 검사한다."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import rospy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, PointCloud2
from tf2_msgs.msg import TFMessage
from validator_msgs.msg import DrivingFeature

from validator_inference.engine import InferenceEngine
from validator_inference.input_buffer import (
    InferenceFrame,
    InputBuffer,
    NotReady,
)
from validator_inference.models.contracts import ModelPrediction
from validator_inference.worker import Completion, SubmitResult


def _stamp(nanoseconds):
    return rospy.Time(
        nanoseconds // 1_000_000_000, nanoseconds % 1_000_000_000
    )


def _profile(
    model_id="diffusiondrive_v2",
    compatibility_flags=(),
    reference_stamp_policy=None,
):
    inference_policy = {}
    if reference_stamp_policy is not None:
        inference_policy["reference_stamp_policy"] = reference_stamp_policy
    return SimpleNamespace(
        model_id=model_id,
        profile_sha256="1" * 64,
        calibration_profile_sha256="2" * 64 if model_id == "sparsedrive_v2" else "",
        data={
            "id": model_id,
            "required_inputs": [
                {
                    "key": "camera_main",
                    "topic": "/image/model",
                    "ros_type": "sensor_msgs/Image",
                },
                {
                    "key": "driving_feature",
                    "topic": "/driving_feature",
                    "ros_type": "validator_msgs/DrivingFeature",
                },
            ],
            "buffer_requirements": {
                "latest_inputs": ["camera_main", "driving_feature"],
                "history_inputs": [],
            },
            "output": {"coordinate_frame": "NAVSIM ego-relative SE2"},
            "static_transforms": {
                "T_base_from_lidar": {
                    "parent_frame_id": "base_link",
                    "child_frame_id": "Lidar3D-4",
                    "translation_m": [0.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            },
            "compatibility_flags": list(compatibility_flags),
            "inference_policy": inference_policy,
        },
    )


def _frame(reference_ns=3_000_000_000, age_ms=25.0):
    return InferenceFrame(
        reference_stamp=_stamp(reference_ns),
        inputs={"camera_main": object(), "driving_feature": object()},
        input_age_ms={"camera_main": age_ms, "driving_feature": 0.0},
        history_count=0,
        history_span_ms=0.0,
    )


class _Publisher:
    def __init__(self, name, actions=None):
        self.name = name
        self.messages = []
        self.actions = actions

    def publish(self, message):
        self.messages.append(message)
        if self.actions is not None:
            self.actions.append("publish:" + self.name)


class _Buffer:
    def __init__(self, result, reference_stamp=None):
        self.result = result
        self.reference_stamp = reference_stamp or _stamp(3_000_000_000)
        self.snapshot_calls = []

    def latest_driving_feature_stamp(self):
        return self.reference_stamp

    def snapshot_at(self, reference_stamp, required_keys):
        self.snapshot_calls.append((reference_stamp, list(required_keys)))
        return self.result


class _Worker:
    def __init__(self, submit_result=None, completion=None, actions=None):
        self.submit_result = submit_result or SubmitResult(True, "accepted")
        self.completion = completion
        self.submitted = []
        self.actions = actions

    def take_completed(self):
        if self.actions is not None:
            self.actions.append("take_completed")
        completion, self.completion = self.completion, None
        return completion

    def try_submit(self, frame):
        self.submitted.append(frame)
        if self.actions is not None:
            self.actions.append("try_submit")
        return self.submit_result


def _engine(buffer, worker, *, profile=None, actions=None):
    metric = _Publisher("metric", actions)
    path = _Publisher("path", actions)
    engine = InferenceEngine(
        selected_profile=profile or _profile(),
        input_buffer=buffer,
        worker=worker,
        path_builder=__import__(
            "validator_inference.path_builder", fromlist=["PathMessageBuilder"]
        ).PathMessageBuilder(
            _profile().data["static_transforms"]["T_base_from_lidar"]
        ),
        path_publisher=path,
        metric_publisher=metric,
        run_id="run-20260721",
    )
    return engine, path, metric


def test_not_ready_yields_exactly_one_metric_with_missing_and_negative_one():
    result = NotReady(
        ["camera_main"],
        input_age_ms={"driving_feature": 0.0},
        history_count=0,
        history_span_ms=0.0,
    )
    engine, path, metric = _engine(_Buffer(result), _Worker())

    engine.on_timer_tick(None)

    assert path.messages == []
    assert len(metric.messages) == 1
    message = metric.messages[0]
    assert message.outcome == "not_ready"
    assert message.elapsed_ms == -1.0
    assert message.missing_input_keys == ["camera_main"]
    assert message.input_keys == ["driving_feature"]
    assert message.input_age_ms == [0.0]
    assert message.tick_index == 1
    assert message.header.stamp == _stamp(3_000_000_000)


def test_busy_yields_one_metric_without_waiting_and_preserves_stale_age():
    frame = _frame(age_ms=12_345.0)
    worker = _Worker(SubmitResult(False, "busy_tick_skipped"))
    engine, path, metric = _engine(_Buffer(frame), worker)

    engine.on_timer_tick(None)

    assert path.messages == []
    assert len(worker.submitted) == 1
    assert len(metric.messages) == 1
    message = metric.messages[0]
    assert message.outcome == "busy_tick_skipped"
    assert message.elapsed_ms == -1.0
    assert dict(zip(message.input_keys, message.input_age_ms)) == {
        "camera_main": 12_345.0,
        "driving_feature": 0.0,
    }
    assert message.missing_input_keys == []


def test_completion_publishes_one_lidar_path_and_one_metric_with_original_tick():
    frame = _frame(reference_ns=7_000_000_009)
    worker = _Worker()
    engine, path, metric = _engine(_Buffer(frame), worker)

    engine.on_timer_tick(None)
    submitted = worker.submitted[0]
    worker.completion = Completion(
        frame=submitted,
        model_id="diffusiondrive_v2",
        result=ModelPrediction([[3.0, 2.0]], "test.source"),
        elapsed_ms=12.5,
        rng_seed=77,
        error=None,
    )

    engine.on_timer_tick(None)

    assert len(path.messages) == 1
    assert path.messages[0].header.frame_id == "Lidar3D-4"
    assert path.messages[0].header.stamp == _stamp(7_000_000_009)
    assert len(metric.messages) == 1
    message = metric.messages[0]
    assert message.outcome == "inference_completed"
    assert message.tick_index == 1
    assert message.header.stamp == _stamp(7_000_000_009)
    assert message.elapsed_ms == 12.5
    assert message.rng_seed == 77
    assert message.run_id == "run-20260721"
    assert message.model_id == "diffusiondrive_v2"
    assert message.profile_sha256 == "1" * 64
    assert message.calibration_profile_sha256 == ""
    assert not hasattr(engine, "time_publisher")


def test_prior_completion_is_published_before_current_frame_submit():
    actions = []
    first_frame = _frame()
    worker = _Worker(actions=actions)
    engine, _, _ = _engine(_Buffer(first_frame), worker, actions=actions)
    engine.on_timer_tick(None)
    worker.completion = Completion(
        frame=worker.submitted[0],
        model_id="diffusiondrive_v2",
        result=ModelPrediction([[1.0, 0.0]], "test.source"),
        elapsed_ms=1.0,
        rng_seed=1,
        error=None,
    )
    actions.clear()

    engine.on_timer_tick(None)

    assert actions.index("publish:metric") < actions.index("try_submit")


def test_new_stamp_once_does_not_resubmit_the_same_reference():
    frame = _frame()
    worker = _Worker()
    engine, _, metric = _engine(
        _Buffer(frame),
        worker,
        profile=_profile(reference_stamp_policy="new_stamp_once"),
    )

    engine.on_timer_tick(None)
    engine.on_timer_tick(None)

    assert worker.submitted == [frame]
    assert metric.messages == []


def test_new_stamp_once_retries_busy_then_accepts_later_stamp():
    first = _frame()
    buffer = _Buffer(first)
    worker = _Worker(SubmitResult(False, "busy_tick_skipped"))
    engine, _, metric = _engine(
        buffer,
        worker,
        profile=_profile(reference_stamp_policy="new_stamp_once"),
    )

    engine.on_timer_tick(None)
    worker.submit_result = SubmitResult(True, "accepted")
    engine.on_timer_tick(None)
    engine.on_timer_tick(None)

    second = _frame(reference_ns=4_000_000_000)
    buffer.result = second
    buffer.reference_stamp = second.reference_stamp
    engine.on_timer_tick(None)

    assert worker.submitted == [first, first, second]
    assert [message.outcome for message in metric.messages] == [
        "busy_tick_skipped"
    ]


def test_rejects_unknown_reference_stamp_policy():
    with pytest.raises(ValueError, match="reference_stamp_policy"):
        _engine(
            _Buffer(_frame()),
            _Worker(),
            profile=_profile(reference_stamp_policy="queue_forever"),
        )


def test_metric_uses_only_profile_declared_compatibility_flags():
    profile = _profile(compatibility_flags=("camera_rig_differs_from_training",))
    engine, _, metric = _engine(
        _Buffer(NotReady(["camera_main"])), _Worker(), profile=profile
    )

    engine.on_timer_tick(None)

    assert metric.messages[0].compatibility_flags == [
        "camera_rig_differs_from_training"
    ]


def test_tfv6_rejects_synthetic_radar_compatibility_flag():
    profile = _profile("tfv6", ("synthetic_radar_empty",))
    with pytest.raises(ValueError, match="synthetic radar"):
        _engine(_Buffer(NotReady([])), _Worker(), profile=profile)


class _SubscriberFactory:
    def __init__(self):
        self.calls = []

    def __call__(self, topic, ros_type, callback, queue_size):
        subscription = SimpleNamespace(
            topic=topic,
            ros_type=ros_type,
            callback=callback,
            queue_size=queue_size,
        )
        self.calls.append(subscription)
        return subscription


def test_input_buffer_configure_owns_one_subscriber_per_declared_input():
    factory = _SubscriberFactory()
    selected = _profile()

    buffer = InputBuffer.configure(selected, subscriber_factory=factory)

    assert [(call.topic, call.ros_type) for call in factory.calls] == [
        ("/image/model", Image),
        ("/driving_feature", DrivingFeature),
    ]
    assert buffer.subscribers == tuple(factory.calls)

    feature = DrivingFeature()
    feature.header.stamp = _stamp(5)
    factory.calls[1].callback(feature)
    assert buffer.latest_driving_feature_stamp() == _stamp(5)


def test_tfv6_configure_adds_tf_pose_callback_without_duplicate_lidar_subscriber():
    factory = _SubscriberFactory()
    profile = SimpleNamespace(
        data={
            "required_inputs": [
                {
                    "key": "lidar_raw",
                    "topic": "/lidar3D",
                    "ros_type": "sensor_msgs/PointCloud2",
                },
                {
                    "key": "driving_feature",
                    "topic": "/driving_feature",
                    "ros_type": "validator_msgs/DrivingFeature",
                },
            ],
            "buffer_requirements": {
                "latest_inputs": ["driving_feature"],
                "history_inputs": [
                    {
                        "key": "lidar_raw",
                        "topic": "/lidar3D",
                        "ros_type": "sensor_msgs/PointCloud2",
                        "fifo_depth": 10,
                        "max_tf_bracket_gap_ms": 120,
                    }
                ],
            },
        }
    )

    buffer = InputBuffer.configure(profile, subscriber_factory=factory)

    assert [(call.topic, call.ros_type) for call in factory.calls] == [
        ("/lidar3D", PointCloud2),
        ("/driving_feature", DrivingFeature),
        ("/tf", TFMessage),
    ]
    transform = TransformStamped()
    transform.header.stamp = _stamp(10)
    transform.header.frame_id = "/map"
    transform.child_frame_id = "/base_link"
    transform.transform.translation.x = 4.0
    transform.transform.rotation.w = 1.0
    factory.calls[2].callback(TFMessage([transform]))
    pose = buffer.pose_histories["lidar_raw"].interpolate(_stamp(10))
    assert pose.translation_xyz == (4.0, 0.0, 0.0)


def test_tf_callback_storage_is_bounded():
    factory = _SubscriberFactory()
    profile = SimpleNamespace(
        data={
            "required_inputs": [
                {
                    "key": "lidar_raw",
                    "topic": "/lidar3D",
                    "ros_type": "sensor_msgs/PointCloud2",
                }
            ],
            "buffer_requirements": {
                "latest_inputs": [],
                "history_inputs": [
                    {
                        "key": "lidar_raw",
                        "fifo_depth": 10,
                        "max_tf_bracket_gap_ms": 120,
                    }
                ],
            },
        }
    )
    buffer = InputBuffer.configure(profile, subscriber_factory=factory)
    tf_callback = factory.calls[-1].callback

    for index in range(300):
        transform = TransformStamped()
        transform.header.stamp = _stamp(index + 1)
        transform.header.frame_id = "map"
        transform.child_frame_id = "base_link"
        transform.transform.rotation.w = 1.0
        tf_callback(TFMessage([transform]))

    assert buffer.pose_histories["lidar_raw"].sample_count == 256


def test_input_buffer_rejects_undeclared_ros_type_before_subscribing():
    factory = _SubscriberFactory()
    profile = SimpleNamespace(
        data={
            "required_inputs": [
                {"key": "bad", "topic": "/bad", "ros_type": "std_msgs/String"}
            ],
            "buffer_requirements": {"latest_inputs": ["bad"], "history_inputs": []},
        }
    )

    with pytest.raises(ValueError, match="unsupported inference input ROS type"):
        InputBuffer.configure(profile, subscriber_factory=factory)
    assert factory.calls == []


def _load_node_script():
    script = Path(__file__).resolve().parents[1] / "scripts/inference_engine_node.py"
    spec = spec_from_file_location("inference_engine_node", script)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, script.read_text(encoding="utf-8")


def test_node_requires_sim_time_and_declares_exactly_one_tenth_second_timer():
    module, source = _load_node_script()

    with pytest.raises(RuntimeError, match="use_sim_time"):
        module.require_sim_time(lambda name, default: False)
    module.require_sim_time(lambda name, default: True)

    assert source.count("rospy.Timer(") == 1
    assert "rospy.Duration.from_sec(0.1)" in source
    assert "rospy.Subscriber(" not in source
    assert "/inference_time" not in source


def test_node_passes_validated_environment_project_root_to_model_selection(
    tmp_path, monkeypatch
):
    """wrapper가 검증한 project root를 concrete model asset 해석에 사용한다."""
    module, _source = _load_node_script()
    selected_profile = object()
    calls = []
    project_root = tmp_path.resolve()
    monkeypatch.setenv("OPENLOOP_PROJECT_ROOT", str(project_root))

    def record_build(profile, **kwargs):
        """model selector 호출 형태를 기록한다."""
        calls.append((profile, kwargs))
        return "model"

    monkeypatch.setattr(
        module,
        "build_selected_model",
        record_build,
    )

    assert module._build_model_for_project(selected_profile) == "model"
    assert calls == [(selected_profile, {"project_root": project_root})]

    monkeypatch.delenv("OPENLOOP_PROJECT_ROOT")
    calls.clear()
    assert module._build_model_for_project(selected_profile) == "model"
    assert calls == [(selected_profile, {})]


@pytest.mark.parametrize("value", ("relative", "/tmp/../tmp", "/missing/project"))
def test_node_rejects_unvalidated_environment_project_root(value, monkeypatch):
    """직접 node 실행도 상대·비정규·누락 project root를 신뢰하지 않는다."""
    module, _source = _load_node_script()
    monkeypatch.setenv("OPENLOOP_PROJECT_ROOT", value)

    with pytest.raises(RuntimeError, match="OPENLOOP_PROJECT_ROOT"):
        module._build_model_for_project(object())


def test_catkin_contract_installs_engine_and_declares_direct_tf_message_dependency():
    package_root = Path(__file__).resolve().parents[1]
    cmake = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "catkin_install_python(PROGRAMS scripts/inference_engine_node.py" in cmake
    assert "tf2_msgs" in cmake
    assert "<build_depend>tf2_msgs</build_depend>" in package_xml
    assert "<exec_depend>tf2_msgs</exec_depend>" in package_xml
