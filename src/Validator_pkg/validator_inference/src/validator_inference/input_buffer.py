"""
inference 입력 토픽을 bounded FIFO에 저장하고 causal snapshot과 TF pose를 구성한다.
입력: profile별 perception topic, /driving_feature, 필요한 /route/target_point*, /tf.
출력(pub): 없음. InferenceFrame/NotReady를 engine에 반환.
구독(sub): 위 입력 토픽을 이 객체가 단독 소유한다.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from validator_inference.quaternion_math import quaternion_slerp


@dataclass(frozen=True)
class InputSpec:
    """하나의 inference 입력 topic과 FIFO 요구량을 선언한다.

    입력: 내부 key, ROS topic/type, FIFO depth, 선택적 history count.
    반환: ``InputBuffer`` 초기화에 쓰이는 불변 입력 명세.
    """

    key: str
    topic: str
    ros_type: str
    fifo_depth: int
    history_count: int = 1


@dataclass(frozen=True)
class StampedSample:
    """수신 시각과 원본 ROS message를 함께 보관하는 FIFO 항목이다.

    입력: nanosecond stamp와 원본 message.
    반환: causal snapshot 선택에 쓰이는 불변 sample.
    """

    stamp_ns: int
    message: Any


@dataclass(frozen=True)
class InferenceFrame:
    """한 model forward에 전달할 causal input snapshot이다.

    입력: 기준 stamp, 선택된 inputs, age, LiDAR history 통계.
    반환: worker가 그대로 처리하는 불변 frame.
    """

    reference_stamp: Any
    inputs: Dict[str, Any]
    input_age_ms: Dict[str, float]
    history_count: int
    history_span_ms: float


@dataclass(frozen=True)
class NotReady:
    """기준 시각까지 필요한 입력이 충족되지 않았음을 설명한다.

    입력: 빠진 input key와 확보된 age·history 통계.
    반환: engine이 ``not_ready`` metric으로 바꾸는 불변 결과.
    """

    missing_input_keys: List[str]
    input_age_ms: Dict[str, float] = field(default_factory=dict)
    history_count: int = 0
    history_span_ms: float = 0.0


@dataclass(frozen=True)
class PoseSample:
    """``map``에서 ``base_link``로의 한 시각 pose를 보관한다.

    입력: nanosecond stamp, translation, ``xyzw`` quaternion.
    반환: point-cloud sweep 정렬에 쓰이는 불변 pose.
    """

    stamp_ns: int
    translation_xyz: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]


@dataclass(frozen=True)
class AlignedPointCloudSweep:
    """timestamp가 있는 cloud와 최신 sweep으로의 rigid alignment 쌍이다.

    입력: 원본 ``PointCloud2``와 anchor LiDAR frame으로의 4×4 transform.
    반환: TFv6 temporal preprocessing에 전달할 불변 sweep.
    """

    point_cloud: Any
    T_anchor_lidar_from_sweep_lidar: np.ndarray


def _stamp_ns(stamp: Any) -> int:
    """ROS time 호환 값을 비교 가능한 정수 nanosecond로 변환한다.

    입력: ``to_nsec`` 또는 초·나노초 field를 제공하는 stamp.
    반환: 정수 nanosecond timestamp.
    """
    to_nsec = getattr(stamp, "to_nsec", None)
    if callable(to_nsec):
        return int(to_nsec())

    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    raise TypeError("stamp must provide to_nsec() or integer second/nanosecond fields")


def _message_stamp_ns(message: Any) -> int:
    """ROS message header의 stamp를 정수 nanosecond로 읽는다.

    입력: ``header.stamp``가 있는 ROS message.
    반환: message의 생성 timestamp.
    """
    try:
        return _stamp_ns(message.header.stamp)
    except AttributeError as error:
        raise TypeError("message must provide header.stamp") from error


def _ros_message_types() -> Dict[str, Any]:
    """profile 계약이 허용한 ROS message class만 import한다.

    입력: 없음.
    반환: profile의 ``ros_type`` 문자열과 ROS class의 대응표.
    """
    from geometry_msgs.msg import PointStamped
    from sensor_msgs.msg import Image, PointCloud2
    from tf2_msgs.msg import TFMessage
    from validator_msgs.msg import DrivingFeature

    return {
        "sensor_msgs/Image": Image,
        "sensor_msgs/PointCloud2": PointCloud2,
        "geometry_msgs/PointStamped": PointStamped,
        "validator_msgs/DrivingFeature": DrivingFeature,
        "tf2_msgs/TFMessage": TFMessage,
    }


class InputBuffer:
    """모든 inference subscription을 소유하고 bounded causal FIFO를 만든다.

    입력: profile에서 파생한 ``InputSpec``과 선택적 TF pose history.
    반환: main loop가 소비할 ``InferenceFrame`` 또는 ``NotReady``.
    """

    def __init__(
        self,
        specs: Iterable[InputSpec],
        *,
        pose_histories: Optional[Dict[str, "TfPoseHistory"]] = None,
    ) -> None:
        """입력별 FIFO와 pose history를 검증해 빈 buffer를 초기화한다.

        입력: 고유 key를 가진 input 명세와 선택적 pose history mapping.
        부작용: thread-safe queue, lock, subscriber 보관 공간을 생성한다.
        """
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
        self._pose_histories = dict(pose_histories or {})
        unknown_pose_keys = set(self._pose_histories) - set(self._specs)
        if unknown_pose_keys:
            raise ValueError(
                "pose history keys must name configured inputs: "
                + ", ".join(sorted(unknown_pose_keys))
            )
        self._lock = RLock()
        self._subscribers: Tuple[Any, ...] = ()

    @classmethod
    def configure(
        cls,
        profile: Any,
        *,
        subscriber_factory: Any = None,
    ) -> "InputBuffer":
        """profile FIFO를 만들고 모든 inference-layer subscription을 소유한다.

        입력: 선택 profile과 선택적 ROS subscriber factory.
        반환: 각 required topic 및 필요한 ``/tf``를 구독하는 ``InputBuffer``.
        """
        data = getattr(profile, "data", profile)
        required = list(data.get("required_inputs", ()))
        requirements = data.get("buffer_requirements", {})
        history_by_key = {
            item["key"]: item
            for item in requirements.get("history_inputs", ())
        }

        message_types = _ros_message_types()
        specs: List[InputSpec] = []
        seen_topics = set()
        for declaration in required:
            key = str(declaration["key"])
            topic = str(declaration["topic"])
            ros_type = str(declaration["ros_type"])
            if ros_type not in message_types or ros_type == "tf2_msgs/TFMessage":
                raise ValueError(
                    "unsupported inference input ROS type: {}".format(ros_type)
                )
            if topic in seen_topics:
                raise ValueError(
                    "duplicate inference input subscription topic: {}".format(topic)
                )
            seen_topics.add(topic)
            history = history_by_key.get(key)
            fifo_depth = int(history["fifo_depth"]) if history else 2
            history_count = fifo_depth if history else 1
            specs.append(
                InputSpec(key, topic, ros_type, fifo_depth, history_count)
            )

        unknown_history = set(history_by_key) - {spec.key for spec in specs}
        if unknown_history:
            raise ValueError(
                "history input is not declared in required_inputs: "
                + ", ".join(sorted(unknown_history))
            )

        pose_histories = {
            key: TfPoseHistory(float(item["max_tf_bracket_gap_ms"]))
            for key, item in history_by_key.items()
            if item.get("paired_pose") or key == "lidar_raw"
        }
        buffer = cls(specs, pose_histories=pose_histories)

        if subscriber_factory is None:
            import rospy

            subscriber_factory = rospy.Subscriber

        subscribers = []
        for spec in specs:
            callback = lambda message, key=spec.key: buffer.store(key, message)
            subscribers.append(
                subscriber_factory(
                    spec.topic,
                    message_types[spec.ros_type],
                    callback,
                    queue_size=1,
                )
            )
        if pose_histories:
            subscribers.append(
                subscriber_factory(
                    "/tf",
                    message_types["tf2_msgs/TFMessage"],
                    buffer._store_tf_message,
                    queue_size=1,
                )
            )
        buffer._subscribers = tuple(subscribers)
        return buffer

    @property
    def subscribers(self) -> Tuple[Any, ...]:
        """buffer가 소유해 수명 동안 유지해야 할 ROS subscriber를 반환한다.

        입력: 없음.
        반환: profile 입력 topic과 선택적 ``/tf`` subscriber tuple.
        """
        return self._subscribers

    @property
    def pose_histories(self) -> Dict[str, "TfPoseHistory"]:
        """history input별 TF pose history의 방어적 mapping 복사본을 반환한다.

        입력: 없음.
        반환: input key에서 ``TfPoseHistory``로의 mapping.
        """
        return dict(self._pose_histories)

    def latest_driving_feature_stamp(self) -> Optional[Any]:
        """수신된 driving feature 중 가장 최신 기준 시각을 반환한다.

        입력: 없음.
        반환: 기준 stamp 또는 driving feature가 없을 때 ``None``.
        """
        with self._lock:
            samples = self._samples.get("driving_feature", ())
            if not samples:
                return None
            newest = max(samples, key=lambda sample: sample.stamp_ns)
            return newest.message.header.stamp

    def _store_tf_message(self, message: Any) -> None:
        """``/tf``에서 map-to-base transform만 골라 모든 history에 저장한다.

        입력: ``tf2_msgs/TFMessage``.
        부작용: 유효한 ``map → base_link`` pose를 timestamp 순서로 추가한다.
        """
        for transform in message.transforms:
            parent = str(transform.header.frame_id).lstrip("/")
            child = str(transform.child_frame_id).lstrip("/")
            if parent != "map" or child != "base_link":
                continue
            value = transform.transform
            pose = PoseSample(
                _stamp_ns(transform.header.stamp),
                (
                    float(value.translation.x),
                    float(value.translation.y),
                    float(value.translation.z),
                ),
                (
                    float(value.rotation.x),
                    float(value.rotation.y),
                    float(value.rotation.z),
                    float(value.rotation.w),
                ),
            )
            for history in self._pose_histories.values():
                history.add(pose)

    def store(self, key: str, message: Any) -> None:
        """subscription callback이 전달한 message를 해당 FIFO의 끝에 저장한다.

        입력: profile input key와 ``header.stamp``가 있는 ROS message.
        부작용: depth를 넘은 가장 오래된 sample을 버린다.
        """
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
        """기준 시각 이후 sample을 배제한 causal model input snapshot을 만든다.

        입력: driving feature 기준 stamp와 모델이 요구하는 input key 순서.
        반환: pose-align된 history를 포함한 ``InferenceFrame`` 또는 ``NotReady``.
        """
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
                        if causal:
                            newest = causal[-1]
                            input_age_ms[key] = _milliseconds(
                                reference_ns - newest.stamp_ns
                            )
                            history_count = max(history_count, len(causal))
                            history_span_ns = max(
                                history_span_ns,
                                causal[-1].stamp_ns - causal[0].stamp_ns,
                            )
                        missing.append(f"{key} history({spec.history_count})")
                        continue
                    selected = causal[-spec.history_count :]
                    newest = selected[-1]
                    input_age_ms[key] = _milliseconds(
                        reference_ns - newest.stamp_ns
                    )
                    history_count = len(selected)
                    history_span_ns = selected[-1].stamp_ns - selected[0].stamp_ns
                    pose_history = self._pose_histories.get(key)
                    if pose_history is None:
                        if spec.ros_type == "sensor_msgs/PointCloud2":
                            missing.append(f"{key} pose_history")
                            continue
                        history_inputs = [sample.message for sample in selected]
                    else:
                        poses = [
                            pose_history.interpolate(sample.message.header.stamp)
                            for sample in selected
                        ]
                        missing_pose_stamps = [
                            sample.stamp_ns
                            for sample, pose in zip(selected, poses)
                            if pose is None
                        ]
                        if missing_pose_stamps:
                            missing.extend(
                                f"{key} pose@{stamp_ns}"
                                for stamp_ns in missing_pose_stamps
                            )
                            continue
                        anchor_from_map = np.linalg.inv(_pose_matrix(poses[-1]))
                        history_inputs = [
                            AlignedPointCloudSweep(
                                sample.message,
                                anchor_from_map @ _pose_matrix(pose),
                            )
                            for sample, pose in zip(selected, poses)
                        ]
                    inputs[f"{key}_history"] = history_inputs
                    continue

                if not causal:
                    missing.append(key)
                    continue
                selected = causal[-1]
                inputs[key] = selected.message
                input_age_ms[key] = _milliseconds(reference_ns - selected.stamp_ns)

            if missing:
                return NotReady(
                    list(missing),
                    input_age_ms=dict(input_age_ms),
                    history_count=history_count,
                    history_span_ms=_milliseconds(history_span_ns),
                )

            return InferenceFrame(
                reference_stamp=reference_stamp,
                inputs=dict(inputs),
                input_age_ms=dict(input_age_ms),
                history_count=history_count,
                history_span_ms=_milliseconds(history_span_ns),
            )


class TfPoseHistory:
    """bounded-gap interpolation을 제공하는 thread-safe timestamp pose history다.

    입력: 허용할 TF bracket gap과 보관 sample 상한.
    반환: temporal LiDAR sweep을 anchor 시각으로 정렬할 pose.
    """

    def __init__(self, max_bracket_gap_ms: float, max_samples: int = 256) -> None:
        """유한한 시간순 pose 저장소를 초기화한다.

        입력: 보간 가능한 최대 gap과 메모리에 유지할 pose 수.
        부작용: timestamp-ordered pose list와 lock을 생성한다.
        """
        if max_bracket_gap_ms < 0:
            raise ValueError("max_bracket_gap_ms must be nonnegative")
        if max_samples < 2:
            raise ValueError("max_samples must be at least two")
        self._max_bracket_gap_ns = int(max_bracket_gap_ms * 1_000_000)
        self._max_samples = int(max_samples)
        self._poses: List[PoseSample] = []
        self._lock = RLock()

    def add(self, pose: PoseSample) -> None:
        """pose를 timestamp 순서에 맞춰 삽입하거나 같은 시각 값을 교체한다.

        입력: ``map → base_link`` ``PoseSample``.
        부작용: 최대 sample 수를 초과한 가장 오래된 pose를 제거한다.
        """
        with self._lock:
            stamps = [sample.stamp_ns for sample in self._poses]
            index = bisect_left(stamps, pose.stamp_ns)
            if index < len(self._poses) and self._poses[index].stamp_ns == pose.stamp_ns:
                self._poses[index] = pose
            else:
                self._poses.insert(index, pose)
            if len(self._poses) > self._max_samples:
                del self._poses[: len(self._poses) - self._max_samples]

    @property
    def sample_count(self) -> int:
        """현재 interpolation에 사용할 수 있는 pose 수를 반환한다.

        입력: 없음.
        반환: 보관 중인 ``PoseSample`` 개수.
        """
        with self._lock:
            return len(self._poses)

    def interpolate(self, stamp: Any) -> Optional[PoseSample]:
        """요청 시각을 양쪽 TF pose 사이에서 선형·slerp 보간한다.

        입력: 보간할 ROS time 호환 stamp.
        반환: gap 계약을 만족한 pose, 범위 밖이면 ``None``.
        """
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
    """정수 nanosecond duration을 metric용 millisecond로 바꾼다.

    입력: nanosecond 단위 duration.
    반환: float millisecond duration.
    """
    return nanoseconds / 1_000_000.0


def _pose_matrix(pose: PoseSample) -> np.ndarray:
    """interpolated pose 하나의 ``T_map_from_base`` homogeneous matrix를 만든다.

    입력: translation과 quaternion이 채워진 ``PoseSample``.
    반환: ``map → base_link`` 4×4 float64 transform.
    """
    x, y, z, w = pose.quaternion_xyzw
    norm = x * x + y * y + z * z + w * w
    if norm <= np.finfo(float).eps:
        raise ValueError("pose quaternion must be nonzero")
    scale = 2.0 / norm
    matrix = np.asarray(
        [
            [
                1.0 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
                0.0,
            ],
            [
                scale * (x * y + z * w),
                1.0 - scale * (x * x + z * z),
                scale * (y * z - x * w),
                0.0,
            ],
            [
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1.0 - scale * (x * x + y * y),
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    matrix[:3, 3] = pose.translation_xyz
    return matrix
