"""Compose causal snapshots, the one-slot worker, and ROS outcome messages."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import rospy
from validator_msgs.msg import InferenceMetric

from validator_inference.input_buffer import NotReady


_OUTPUT_FRAME = "Lidar3D-4"


def _profile_flags(selected_profile: Any) -> List[str]:
    data = selected_profile.data
    declared = list(data.get("compatibility_flags", ()))
    rig = data.get("rig_conditions", {})
    declared.extend(rig.get("compatibility_flags", ()))
    flags = list(dict.fromkeys(str(flag) for flag in declared))
    if selected_profile.model_id == "tfv6" and any(
        "synthetic_radar" in flag for flag in flags
    ):
        raise ValueError(
            "official TFv6 no-radar profile cannot declare a synthetic radar flag"
        )
    return flags


class InferenceEngine:
    """Publish exactly one eventual metric for each inference tick outcome."""

    def __init__(
        self,
        *,
        selected_profile: Any,
        input_buffer: Any,
        worker: Any,
        path_builder: Any,
        path_publisher: Any,
        metric_publisher: Any,
        run_id: str,
    ) -> None:
        self._profile = selected_profile
        self._buffer = input_buffer
        self._worker = worker
        self._path_builder = path_builder
        self.path_publisher = path_publisher
        self.metric_publisher = metric_publisher
        self._run_id = str(run_id)
        self._required_keys = [
            str(item["key"])
            for item in selected_profile.data.get("required_inputs", ())
        ]
        self._coordinate_convention = str(
            selected_profile.data["output"]["coordinate_frame"]
        )
        self._compatibility_flags = _profile_flags(selected_profile)
        self._tick_index = 0
        self._submitted_ticks: Dict[int, int] = {}

    @property
    def tick_index(self) -> int:
        return self._tick_index

    def on_timer_tick(self, _event: Any) -> None:
        self._tick_index += 1

        completion = self._worker.take_completed()
        if completion is not None:
            original_tick = self._submitted_ticks.pop(id(completion.frame), None)
            if original_tick is None:
                raise RuntimeError("worker completion has no submitted tick context")
            self._publish_completion(completion, original_tick)

        reference_stamp = self._buffer.latest_driving_feature_stamp()
        if reference_stamp is None:
            self._publish_metric(
                outcome="not_ready",
                tick_index=self._tick_index,
                reference_stamp=rospy.Time(0),
                input_age_ms={},
                missing_input_keys=["driving_feature"],
                history_count=0,
                history_span_ms=0.0,
                elapsed_ms=-1.0,
                rng_seed=0,
            )
            return

        snapshot = self._buffer.snapshot_at(reference_stamp, self._required_keys)
        if isinstance(snapshot, NotReady):
            self._publish_metric(
                outcome="not_ready",
                tick_index=self._tick_index,
                reference_stamp=reference_stamp,
                input_age_ms=snapshot.input_age_ms,
                missing_input_keys=snapshot.missing_input_keys,
                history_count=snapshot.history_count,
                history_span_ms=snapshot.history_span_ms,
                elapsed_ms=-1.0,
                rng_seed=0,
            )
            return

        submitted = self._worker.try_submit(snapshot)
        if not submitted.accepted:
            self._publish_metric(
                outcome="busy_tick_skipped",
                tick_index=self._tick_index,
                reference_stamp=reference_stamp,
                input_age_ms=snapshot.input_age_ms,
                missing_input_keys=[],
                history_count=snapshot.history_count,
                history_span_ms=snapshot.history_span_ms,
                elapsed_ms=-1.0,
                rng_seed=0,
            )
            return
        self._submitted_ticks[id(snapshot)] = self._tick_index

    def _publish_completion(self, completion: Any, tick_index: int) -> None:
        if completion.model_id != self._profile.model_id:
            raise RuntimeError(
                "worker completion model {} does not match selected {}".format(
                    completion.model_id, self._profile.model_id
                )
            )
        if completion.error is not None:
            raise completion.error
        if completion.result is None:
            raise RuntimeError("worker completion has no prediction")
        frame = completion.frame
        path = self._path_builder.build(
            completion.result.trajectory,
            self._coordinate_convention,
            frame.reference_stamp,
        )
        self.path_publisher.publish(path)
        self._publish_metric(
            outcome="inference_completed",
            tick_index=tick_index,
            reference_stamp=frame.reference_stamp,
            input_age_ms=frame.input_age_ms,
            missing_input_keys=[],
            history_count=frame.history_count,
            history_span_ms=frame.history_span_ms,
            elapsed_ms=float(completion.elapsed_ms),
            rng_seed=int(completion.rng_seed),
        )

    def _publish_metric(
        self,
        *,
        outcome: str,
        tick_index: int,
        reference_stamp: Any,
        input_age_ms: Dict[str, float],
        missing_input_keys: Iterable[str],
        history_count: int,
        history_span_ms: float,
        elapsed_ms: float,
        rng_seed: int,
    ) -> None:
        ordered_keys = [key for key in self._required_keys if key in input_age_ms]
        ordered_keys.extend(
            sorted(set(input_age_ms).difference(ordered_keys))
        )
        metric = InferenceMetric()
        metric.header.stamp = reference_stamp
        metric.header.frame_id = _OUTPUT_FRAME
        metric.run_id = self._run_id
        metric.tick_index = int(tick_index)
        metric.model_id = self._profile.model_id
        metric.outcome = outcome
        metric.elapsed_ms = float(elapsed_ms)
        metric.input_keys = ordered_keys
        metric.input_age_ms = [float(input_age_ms[key]) for key in ordered_keys]
        metric.missing_input_keys = list(missing_input_keys)
        metric.compatibility_flags = list(self._compatibility_flags)
        metric.history_count = int(history_count)
        metric.history_span_ms = float(history_span_ms)
        metric.rng_seed = int(rng_seed)
        metric.profile_sha256 = self._profile.profile_sha256
        metric.calibration_profile_sha256 = (
            self._profile.calibration_profile_sha256
        )
        self.metric_publisher.publish(metric)
