import hashlib
import math
import threading
import time
from types import SimpleNamespace

import pytest
import rospy

from validator_inference.input_buffer import InferenceFrame
from validator_inference.models.contracts import ModelPrediction
from validator_inference.path_builder import PathMessageBuilder
import validator_inference.worker as worker_module
from validator_inference.worker import InferenceWorker, seed_from


def _stamp(nanoseconds):
    return rospy.Time(nanoseconds // 1_000_000_000, nanoseconds % 1_000_000_000)


def _frame(nanoseconds=1_250_000_000):
    return InferenceFrame(
        reference_stamp=_stamp(nanoseconds),
        inputs={},
        input_age_ms={},
        history_count=0,
        history_span_ms=0.0,
    )


class _BlockingModel:
    def __init__(self, model_id="tfv6"):
        self.selected_profile = SimpleNamespace(model_id=model_id)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def forward(self, frame):
        self.calls.append(frame)
        self.entered.set()
        assert self.release.wait(timeout=5.0)
        return ModelPrediction([[1.0, 2.0]], "test.BlockingModel")


class _ImmediateModel:
    def __init__(self, model_id="tfv6"):
        self.selected_profile = SimpleNamespace(model_id=model_id)
        self.calls = []

    def forward(self, frame):
        self.calls.append(frame)
        return ModelPrediction([[1.0, 2.0]], "test.ImmediateModel")


class _DiffusionModel:
    def __init__(self, model_id="diffusiondrive_v2"):
        self.selected_profile = SimpleNamespace(model_id=model_id)
        self.seeds = []

    def forward(self, frame, rng_seed):
        self.seeds.append(rng_seed)
        return ModelPrediction([[1.0, 0.0, 0.0]], "test.DiffusionModel")


class _FailingModel:
    def __init__(self):
        self.selected_profile = SimpleNamespace(model_id="interfuser")

    def forward(self, frame):
        raise ValueError("bad source output")


class _TimedDiffusionModel:
    def __init__(self, error=None):
        self.selected_profile = SimpleNamespace(model_id="diffusiondrive_v2")
        self.error = error
        self.seeds = []

    def forward(self, frame, rng_seed):
        self.seeds.append(rng_seed)
        if self.error is not None:
            raise self.error
        return ModelPrediction([[1.0, 0.0, 0.0]], "test.TimedDiffusionModel")


def _wait_completion(worker, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        completion = worker.take_completed()
        if completion is not None:
            return completion
        time.sleep(0.005)
    raise AssertionError("worker did not complete")


def test_worker_rejects_second_submit_without_queueing_or_replacing_work():
    model = _BlockingModel()
    worker = InferenceWorker(model, "run-a")
    first = _frame(1)
    second = _frame(2)

    accepted = worker.try_submit(first)
    assert accepted.accepted is True
    assert accepted.reason == "accepted"
    assert model.entered.wait(timeout=2.0)

    rejected = worker.try_submit(second)
    assert rejected.accepted is False
    assert rejected.reason == "busy_tick_skipped"
    assert model.calls == [first]

    model.release.set()
    completion = _wait_completion(worker)
    assert completion.frame is first
    assert completion.result.trajectory == [[1.0, 2.0]]
    assert completion.error is None
    assert completion.elapsed_ms >= 0.0
    assert completion.rng_seed == 0
    assert worker.take_completed() is None
    assert model.calls == [first]


def test_worker_preserves_completion_arriving_between_take_and_next_submit():
    """완료 mailbox가 소비될 때까지 새 forward를 막는지 검사한다.

    engine이 완료를 보지 못한 뒤, 같은 timer가 다음 frame을 제출하기 직전에
    첫 forward가 끝나는 interleaving을 재현한다. 여기서 두 번째 frame을 받으면
    한 칸짜리 mailbox의 첫 결과를 덮어쓰게 된다.
    """
    model = _ImmediateModel()
    worker = InferenceWorker(model, "run-mailbox")
    first = _frame(1)
    second = _frame(2)

    # 비결정적인 thread를 예약하지 않고 실행 중인 첫 forward를 모사한 뒤,
    # 정확한 take -> complete -> submit interleaving을 실행한다.
    assert worker._forward_lock.acquire(blocking=False) is True
    assert worker.take_completed() is None
    worker._run_forward(first)

    rejected = worker.try_submit(second)

    assert rejected.accepted is False
    assert rejected.reason == "busy_tick_skipped"
    first_completion = worker.take_completed()
    assert first_completion is not None
    assert first_completion.frame is first

    assert worker.try_submit(second).accepted is True
    second_completion = _wait_completion(worker)
    assert second_completion.frame is second
    assert model.calls == [first, second]


def test_diffusion_seed_is_stable_big_endian_uint64_and_forwarded_only_to_ddv2():
    reference_ns = 9_876_543_210
    expected = int.from_bytes(
        hashlib.sha256(
            b"diffusiondrive_v2:9876543210:run-a"
        ).digest()[:8],
        "big",
    )
    assert seed_from("run-a", "diffusiondrive_v2", reference_ns) == expected
    assert seed_from("run-a", "diffusiondrive_v2", reference_ns) == expected
    assert seed_from("run-b", "diffusiondrive_v2", reference_ns) != expected
    assert seed_from("run-a", "diffusiondrive_v2", reference_ns + 1) != expected
    assert seed_from("run-a", "interfuser", reference_ns) != expected

    model = _DiffusionModel()
    frame = _frame(reference_ns)
    worker = InferenceWorker(model, "run-a")
    assert worker.try_submit(frame).accepted is True
    completion = _wait_completion(worker)
    assert model.seeds == [expected]
    assert completion.rng_seed == expected
    assert completion.result.source_symbol == "test.DiffusionModel"


def test_worker_retains_model_context_on_error_and_releases_busy_state():
    model = _FailingModel()
    worker = InferenceWorker(model, "run-error")
    frame = _frame(11)

    assert worker.try_submit(frame).accepted is True
    completion = _wait_completion(worker)
    assert completion.frame is frame
    assert completion.result is None
    assert completion.model_id == "interfuser"
    assert completion.elapsed_ms >= 0.0
    assert completion.rng_seed == 0
    assert completion.error.model_id == "interfuser"
    assert isinstance(completion.error.__cause__, ValueError)
    assert "bad source output" in str(completion.error)

    assert worker.try_submit(frame).accepted is True
    second_completion = _wait_completion(worker)
    assert second_completion.error is not None


@pytest.mark.parametrize("model_error", [None, ValueError("source failure")])
def test_worker_measures_only_forward_after_diffusion_seed_preparation(
    monkeypatch, model_error
):
    """seed 준비 시간은 clock을 소비하지만 성공·실패 모두 측정에서 제외한다."""
    clock_values = iter(
        [10_000_000, 20_000_000, 100_000_000, 150_000_000]
    )

    monkeypatch.setattr(worker_module, "perf_counter_ns", lambda: next(clock_values))

    def seeded_after_expensive_setup(run_id, model_id, reference_stamp_ns):
        del run_id, model_id, reference_stamp_ns
        worker_module.perf_counter_ns()
        worker_module.perf_counter_ns()
        return 77

    monkeypatch.setattr(worker_module, "seed_from", seeded_after_expensive_setup)
    model = _TimedDiffusionModel(error=model_error)
    worker = InferenceWorker(model, "run-timed")

    assert worker._forward_lock.acquire(blocking=False) is True
    worker._run_forward(_frame(44))
    completion = worker.take_completed()

    assert model.seeds == [77]
    assert completion.rng_seed == 77
    assert completion.elapsed_ms == pytest.approx(50.0)
    if model_error is None:
        assert completion.error is None
        assert completion.result is not None
    else:
        assert completion.result is None
        assert isinstance(completion.error.__cause__, ValueError)


def _transform(translation=(0.0, 0.0, 0.0), quaternion=(0.0, 0.0, 0.0, 1.0)):
    return {
        "parent_frame_id": "base_link",
        "child_frame_id": "Lidar3D-4",
        "translation_m": list(translation),
        "quaternion_xyzw": list(quaternion),
    }


def _yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def test_carla_path_flips_y_then_applies_actual_bag_inverse_transform():
    stamp = _stamp(3_000_000_004)
    builder = PathMessageBuilder(_transform(translation=(2.0, 0.0, 1.5)))

    path = builder.build([[3.0, 2.0]], "CARLA ego-relative", stamp)

    assert path.header.frame_id == "Lidar3D-4"
    assert path.header.stamp == stamp
    assert len(path.poses) == 1
    pose = path.poses[0]
    assert pose.header.frame_id == "Lidar3D-4"
    assert pose.header.stamp == stamp
    assert pose.pose.position.x == pytest.approx(1.0)
    assert pose.pose.position.y == pytest.approx(-2.0)
    assert pose.pose.position.z == pytest.approx(-1.5)
    assert pose.pose.orientation.w == pytest.approx(1.0)


def test_navsim_path_does_not_flip_y():
    path = PathMessageBuilder(_transform()).build(
        [[3.0, 2.0]], "NAVSIM ego-relative SE2", _stamp(1)
    )
    assert path.poses[0].pose.position.x == pytest.approx(3.0)
    assert path.poses[0].pose.position.y == pytest.approx(2.0)
    assert path.poses[0].pose.position.z == pytest.approx(0.0)


def test_path_applies_nonidentity_rotation_to_position_and_heading():
    half_sqrt = math.sqrt(0.5)
    transform = _transform(quaternion=(0.0, 0.0, half_sqrt, half_sqrt))
    path = PathMessageBuilder(transform).build(
        [[1.0, 0.0, math.pi / 2.0]],
        "NAVSIM ego-relative SE2",
        _stamp(9),
    )
    pose = path.poses[0].pose
    assert pose.position.x == pytest.approx(0.0, abs=1e-12)
    assert pose.position.y == pytest.approx(-1.0)
    assert _yaw_from_quaternion(pose.orientation) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "transform,trajectory,convention,error_match",
    [
        (_transform(), [[1.0, 2.0]], "unknown", "unsupported coordinate convention"),
        (_transform(), [], "NAVSIM ego-relative SE2", "trajectory must not be empty"),
        (_transform(), [[1.0]], "NAVSIM ego-relative SE2", "two or three"),
        (_transform(), [[1.0, 2.0, 3.0, 4.0]], "NAVSIM ego-relative SE2", "two or three"),
        (_transform(), [[1.0, float("nan")]], "NAVSIM ego-relative SE2", "finite"),
        (
            _transform(quaternion=(0.0, 0.0, 0.0, 0.0)),
            [[1.0, 2.0]],
            "NAVSIM ego-relative SE2",
            "quaternion",
        ),
        (
            {"translation_m": [0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
            [[1.0, 2.0]],
            "NAVSIM ego-relative SE2",
            "translation",
        ),
    ],
)
def test_path_rejects_unsupported_or_malformed_inputs(
    transform, trajectory, convention, error_match
):
    with pytest.raises(ValueError, match=error_match):
        PathMessageBuilder(transform).build(trajectory, convention, _stamp(1))
