"""Bounded, causal ROS message storage and map-to-base pose interpolation."""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
import os
import sys
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union

try:
    from tf.transformations import quaternion_slerp
except ModuleNotFoundError as error:
    if error.name != "tf":
        raise
    ros_python_path = "/opt/ros/{}/lib/python3/dist-packages".format(
        os.environ.get("ROS_DISTRO", "noetic")
    )
    sys.path.append(ros_python_path)
    from tf.transformations import quaternion_slerp


@dataclass(frozen=True)
class InputSpec:
    key: str
    topic: str
    ros_type: str
    fifo_depth: int
    history_count: int = 1


@dataclass(frozen=True)
class StampedSample:
    stamp_ns: int
    message: Any


@dataclass(frozen=True)
class InferenceFrame:
    reference_stamp: Any
    inputs: Dict[str, Any]
    input_age_ms: Dict[str, float]
    history_count: int
    history_span_ms: float


@dataclass(frozen=True)
class NotReady:
    missing_input_keys: List[str]


@dataclass(frozen=True)
class PoseSample:
    stamp_ns: int
    translation_xyz: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]


def _stamp_ns(stamp: Any) -> int:
    """Return integer nanoseconds from rospy.Time-compatible values."""
    to_nsec = getattr(stamp, "to_nsec", None)
    if callable(to_nsec):
        return int(to_nsec())

    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    raise TypeError("stamp must provide to_nsec() or integer second/nanosecond fields")


def _message_stamp_ns(message: Any) -> int:
    try:
        return _stamp_ns(message.header.stamp)
    except AttributeError as error:
        raise TypeError("message must provide header.stamp") from error


class InputBuffer:
    """Keep bounded input FIFOs and form stable causal snapshots."""

    def __init__(self, specs: Iterable[InputSpec]) -> None:
        spec_list = list(specs)
        if any(spec.fifo_depth < 1 for spec in spec_list):
            raise ValueError("fifo_depth must be positive")
        if any(
            spec.history_count < 1 or spec.history_count > spec.fifo_depth
            for spec in spec_list
        ):
            raise ValueError("history_count must be between one and fifo_depth")

        self._specs = {spec.key: spec for spec in spec_list}
        if len(self._specs) != len(spec_list):
            raise ValueError("input keys must be unique")
        self._samples: Dict[str, Deque[StampedSample]] = {
            key: deque() for key in self._specs
        }
        self._lock = RLock()

    def store(self, key: str, message: Any) -> None:
        if key not in self._specs:
            raise KeyError(key)
        stamp_ns = _message_stamp_ns(message)
        with self._lock:
            queue = self._samples[key]
            queue.append(StampedSample(stamp_ns, message))
            while len(queue) > self._specs[key].fifo_depth:
                queue.popleft()

    def snapshot_at(
        self, reference_stamp: Any, required_keys: List[str]
    ) -> Union[InferenceFrame, NotReady]:
        reference_ns = _stamp_ns(reference_stamp)
        with self._lock:
            unknown_keys = [key for key in required_keys if key not in self._specs]
            if unknown_keys:
                raise KeyError(unknown_keys[0])

            inputs: Dict[str, Any] = {}
            input_age_ms: Dict[str, float] = {}
            missing: List[str] = []
            history_count = 0
            history_span_ns = 0

            for key in required_keys:
                spec = self._specs[key]
                causal = sorted(
                    (
                        sample
                        for sample in self._samples[key]
                        if sample.stamp_ns <= reference_ns
                    ),
                    key=lambda sample: sample.stamp_ns,
                )

                if spec.history_count > 1:
                    if len(causal) < spec.history_count:
                        missing.append(f"{key} history({spec.history_count})")
                        continue
                    selected = causal[-spec.history_count :]
                    inputs[f"{key}_history"] = [
                        sample.message for sample in selected
                    ]
                    newest = selected[-1]
                    input_age_ms[key] = _milliseconds(
                        reference_ns - newest.stamp_ns
                    )
                    history_count = len(selected)
                    history_span_ns = selected[-1].stamp_ns - selected[0].stamp_ns
                    continue

                if not causal:
                    missing.append(key)
                    continue
                selected = causal[-1]
                inputs[key] = selected.message
                input_age_ms[key] = _milliseconds(reference_ns - selected.stamp_ns)

            if missing:
                return NotReady(list(missing))

            return InferenceFrame(
                reference_stamp=reference_stamp,
                inputs=dict(inputs),
                input_age_ms=dict(input_age_ms),
                history_count=history_count,
                history_span_ms=_milliseconds(history_span_ns),
            )


class TfPoseHistory:
    """Thread-safe timestamped pose history with bounded-gap interpolation."""

    def __init__(self, max_bracket_gap_ms: float) -> None:
        if max_bracket_gap_ms < 0:
            raise ValueError("max_bracket_gap_ms must be nonnegative")
        self._max_bracket_gap_ns = int(max_bracket_gap_ms * 1_000_000)
        self._poses: List[PoseSample] = []
        self._lock = RLock()

    def add(self, pose: PoseSample) -> None:
        with self._lock:
            stamps = [sample.stamp_ns for sample in self._poses]
            index = bisect_left(stamps, pose.stamp_ns)
            if index < len(self._poses) and self._poses[index].stamp_ns == pose.stamp_ns:
                self._poses[index] = pose
            else:
                self._poses.insert(index, pose)

    def interpolate(self, stamp: Any) -> Optional[PoseSample]:
        requested_ns = _stamp_ns(stamp)
        with self._lock:
            stamps = [sample.stamp_ns for sample in self._poses]
            index = bisect_left(stamps, requested_ns)
            if index < len(self._poses) and self._poses[index].stamp_ns == requested_ns:
                return self._poses[index]
            if index == 0 or index == len(self._poses):
                return None

            before = self._poses[index - 1]
            after = self._poses[index]
            bracket_ns = after.stamp_ns - before.stamp_ns
            if bracket_ns > self._max_bracket_gap_ns:
                return None

            fraction = (requested_ns - before.stamp_ns) / float(bracket_ns)
            translation = tuple(
                start + fraction * (end - start)
                for start, end in zip(before.translation_xyz, after.translation_xyz)
            )
            quaternion = tuple(
                float(component)
                for component in quaternion_slerp(
                    before.quaternion_xyzw, after.quaternion_xyzw, fraction
                )
            )
            return PoseSample(requested_ns, translation, quaternion)


def _milliseconds(nanoseconds: int) -> float:
    return nanoseconds / 1_000_000.0
