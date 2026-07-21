"""유한 인과 추론 입력 저장소와 TF pose 이력을 검사한다."""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import pytest

from validator_inference.input_buffer import (
    InferenceFrame,
    InputBuffer,
    InputSpec,
    NotReady,
    PoseSample,
    TfPoseHistory,
)


@dataclass(frozen=True)
class FakeStamp:
    secs: int
    nsecs: int

    def to_nsec(self):
        return self.secs * 1_000_000_000 + self.nsecs


@dataclass(frozen=True)
class FakeHeader:
    stamp: FakeStamp


@dataclass(frozen=True)
class FakeMessage:
    value: object
    header: FakeHeader


def time_from_sec(seconds):
    whole_seconds = int(seconds)
    nanoseconds = int(round((seconds - whole_seconds) * 1_000_000_000))
    return FakeStamp(whole_seconds, nanoseconds)


def stamped(value, seconds):
    return FakeMessage(value, FakeHeader(time_from_sec(seconds)))


def transform_at(seconds, x=0.0):
    return PoseSample(
        stamp_ns=time_from_sec(seconds).to_nsec(),
        translation_xyz=(x, 0.0, 0.0),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
    )


def test_snapshot_uses_last_sample_not_later_than_reference():
    buffer = InputBuffer([InputSpec("camera", "/cam", "sensor_msgs/Image", 2)])
    buffer.store("camera", stamped("old", 1.0))
    buffer.store("camera", stamped("future", 3.0))

    frame = buffer.snapshot_at(time_from_sec(2.0), ["camera"])

    assert isinstance(frame, InferenceFrame)
    assert frame.inputs["camera"].value == "old"
    assert frame.input_age_ms["camera"] == 1000.0


def test_stale_data_is_valid_and_records_nonnegative_age():
    buffer = InputBuffer([InputSpec("camera", "/cam", "sensor_msgs/Image", 2)])
    buffer.store("camera", stamped("stale", 1.125))

    frame = buffer.snapshot_at(time_from_sec(3.0), ["camera"])

    assert isinstance(frame, InferenceFrame)
    assert frame.inputs["camera"].value == "stale"
    assert frame.input_age_ms["camera"] == 1875.0


def test_tfv6_requires_exactly_ten_past_sweeps():
    buffer = InputBuffer(
        [
            InputSpec(
                "lidar_raw",
                "/lidar3D",
                "sensor_msgs/PointCloud2",
                10,
                history_count=10,
            )
        ]
    )
    for index in range(9):
        buffer.store("lidar_raw", stamped(index, float(index)))

    result = buffer.snapshot_at(time_from_sec(9.0), ["lidar_raw"])

    assert isinstance(result, NotReady)
    assert result.missing_input_keys == ["lidar_raw history(10)"]


def test_tfv6_ten_sweep_history_is_chronological_with_count_and_span():
    poses = TfPoseHistory(max_bracket_gap_ms=120)
    for index in range(10):
        poses.add(transform_at(index * 0.1))
    buffer = InputBuffer(
        [
            InputSpec(
                "lidar_raw",
                "/lidar3D",
                "sensor_msgs/PointCloud2",
                10,
                history_count=10,
            )
        ],
        pose_histories={"lidar_raw": poses},
    )
    for index in reversed(range(10)):
        buffer.store("lidar_raw", stamped(index, index * 0.1))

    frame = buffer.snapshot_at(time_from_sec(1.0), ["lidar_raw"])

    assert isinstance(frame, InferenceFrame)
    assert [
        sample.point_cloud.value for sample in frame.inputs["lidar_raw_history"]
    ] == list(range(10))
    assert frame.history_count == 10
    assert frame.history_span_ms == 900.0
    assert frame.input_age_ms["lidar_raw"] == 100.0


def test_snapshot_returns_stable_selected_containers_after_more_stores():
    buffer = InputBuffer(
        [InputSpec("lidar_raw", "/lidar3D", "PointCloud2", 2, history_count=2)]
    )
    buffer.store("lidar_raw", stamped("one", 1.0))
    buffer.store("lidar_raw", stamped("two", 2.0))
    frame = buffer.snapshot_at(time_from_sec(2.0), ["lidar_raw"])

    buffer.store("lidar_raw", stamped("three", 3.0))

    assert [sample.value for sample in frame.inputs["lidar_raw_history"]] == [
        "one",
        "two",
    ]


def test_unknown_store_key_is_rejected():
    buffer = InputBuffer([InputSpec("camera", "/cam", "Image", 2)])

    with pytest.raises(KeyError):
        buffer.store("unknown", object())


def test_pose_interpolation_rejects_a_gap_over_profile_limit():
    history = TfPoseHistory(max_bracket_gap_ms=120)
    history.add(transform_at(0.0))
    history.add(transform_at(0.5))

    assert history.interpolate(time_from_sec(0.25)) is None


def test_pose_interpolation_returns_midpoint_translation_and_rotation():
    history = TfPoseHistory(max_bracket_gap_ms=120)
    history.add(transform_at(1.0, x=2.0))
    history.add(transform_at(1.1, x=4.0))

    pose = history.interpolate(time_from_sec(1.05))

    assert pose is not None
    assert pose.stamp_ns == time_from_sec(1.05).to_nsec()
    assert pose.translation_xyz == pytest.approx((3.0, 0.0, 0.0))
    assert pose.quaternion_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_pose_exact_timestamp_does_not_require_a_bracket():
    history = TfPoseHistory(max_bracket_gap_ms=120)
    expected = transform_at(2.0, x=7.0)
    history.add(expected)

    assert history.interpolate(time_from_sec(2.0)) == expected


def test_pose_without_bracketing_pair_is_missing():
    history = TfPoseHistory(max_bracket_gap_ms=120)
    history.add(transform_at(1.0))
    history.add(transform_at(1.1))

    assert history.interpolate(time_from_sec(0.5)) is None
    assert history.interpolate(time_from_sec(1.5)) is None


def test_clean_importing_input_buffer_rejects_tf_dependencies():
    package_source = Path(__file__).parents[1] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(package_source)
    script = dedent(
        """
        import importlib.abc
        import sys

        attempted = []

        class RejectTfImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if (
                    fullname == "tf"
                    or fullname.startswith("tf.")
                    or fullname.startswith("tf2_py")
                ):
                    attempted.append(fullname)
                    raise ImportError("rejected TF dependency: " + fullname)
                return None

        sys.meta_path.insert(0, RejectTfImports())
        import validator_inference.input_buffer
        assert not attempted, attempted
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
