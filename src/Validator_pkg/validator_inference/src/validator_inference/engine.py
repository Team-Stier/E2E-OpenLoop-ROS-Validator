"""
causal snapshot, one-slot worker, ROS outcome metric을 조합하는 inference engine 본체.
입력: InputBuffer snapshot, InferenceWorker completion, 선택된 profile.
출력(pub): /waypoints, /inference_metric.
구독(sub): 없음. 구독은 InputBuffer가 소유한다.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import rospy
from validator_msgs.msg import InferenceMetric

from validator_inference.input_buffer import NotReady


_OUTPUT_FRAME = "Lidar3D-4"


def _profile_flags(selected_profile: Any) -> List[str]:
    """profile과 rig condition의 compatibility flag를 중복 없이 합친다.

    입력: 선택 model profile.
    반환: 기록할 안정된 compatibility flag 목록.
    """
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
    """10 Hz tick마다 causal snapshot·worker·ROS 출력을 조합한다.

    입력: selected profile, input buffer, one-slot worker, path/metric publisher.
    반환: 성공 시 waypoints, 모든 tick에서 하나의 ``InferenceMetric``.
    """

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
        """선택 profile과 협력 객체를 묶어 engine의 tick 상태를 초기화한다.

        입력: buffer, worker, path builder, ROS publishers, 평가 run id.
        부작용: required input 순서와 submitted frame별 원래 tick을 저장한다.
        """
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
        reference_policy = selected_profile.data.get("inference_policy", {}).get(
            "reference_stamp_policy", "every_tick"
        )
        if reference_policy not in {"every_tick", "new_stamp_once"}:
            raise ValueError("unsupported reference_stamp_policy")
        self._new_stamp_once = reference_policy == "new_stamp_once"
        self._last_submitted_reference_ns = None
        self._tick_index = 0
        self._submitted_ticks: Dict[int, int] = {}

    @property
    def tick_index(self) -> int:
        """현재까지 처리한 10 Hz engine tick 번호를 반환한다.

        입력: 없음.
        반환: 1부터 증가하는 현재 tick index.
        """
        return self._tick_index

    def on_timer_tick(self, _event: Any) -> None:
        """timer 한 번에서 completion, not-ready, busy, submit 중 하나를 처리한다.

        입력: ``rospy.Timer`` event(값은 사용하지 않음).
        부작용: waypoints 또는 metric을 발행하고 필요하면 worker에 frame을 제출한다.
        """
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
        reference_ns = reference_stamp.to_nsec()
        if (
            self._new_stamp_once
            and reference_ns == self._last_submitted_reference_ns
        ):
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
        if self._new_stamp_once:
            self._last_submitted_reference_ns = reference_ns
        self._submitted_ticks[id(snapshot)] = self._tick_index

    def _publish_completion(self, completion: Any, tick_index: int) -> None:
        """완료된 model prediction을 LiDAR path와 완료 metric으로 발행한다.

        입력: worker completion과 frame이 최초 제출된 tick 번호.
        부작용: ``/waypoints``와 ``/inference_metric``을 순서대로 발행한다.
        """
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
        """어떤 tick outcome이든 공통 ``InferenceMetric`` 하나로 발행한다.

        입력: outcome, 기준 stamp, input age, history, 측정 시간, RNG seed.
        부작용: profile/calibration hash를 포함한 metric을 publisher로 전송한다.
        """
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
