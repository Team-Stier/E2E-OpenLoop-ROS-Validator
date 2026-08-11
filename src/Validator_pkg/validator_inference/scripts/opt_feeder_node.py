#!/usr/bin/env python3
"""
/Ego_topic을 모델별 non-perception feature로 변환하는 OptFeeder ROS1 노드.
입력: /Ego_topic, 로컬 profile.json.
출력(pub): /driving_feature, 필요한 /route/target_point*.
구독(sub): /Ego_topic, command sidecar replay 시 /clock.
"""

import json
import re
from collections import deque
from pathlib import Path
from threading import Lock

import rospkg
import rospy
from geometry_msgs.msg import PointStamped
from morai_msgs.msg import EgoVehicleStatus
from rosgraph_msgs.msg import Clock
from validator_msgs.msg import CameraCalibration, DrivingFeature

from validator_inference.opt_features import (
    CameraCalibrationData,
    build_driving_feature_data,
    build_fallback_targets,
)
from validator_inference.profile_store import load_selected_profile


_COMMAND_NAMES = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_FIELDS = (
    "domain_id",
    "resolved_domain_sha256",
    "sensor_preset_sha256",
    "source_bag_sha256",
    "command_rule_sha256",
)
_PROFILE_IDENTITY_FIELDS = ("domain_id", "resolved_domain_sha256")


def load_command_sidecar(path, expected_identity):
    """receipt-clock command schedule을 엄격한 순서로 읽는다."""
    schedule = []
    identity = None
    previous_timestamp = None
    if (
        not isinstance(expected_identity.get("domain_id"), str)
        or not expected_identity["domain_id"]
        or not isinstance(expected_identity.get("resolved_domain_sha256"), str)
        or not _SHA256.fullmatch(expected_identity["resolved_domain_sha256"])
    ):
        raise ValueError("invalid command sidecar profile identity")
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("invalid command sidecar row")
            stamp = row.get("grid_receipt_ns")
            command = row.get("command_index")
            if (
                row.get("schema_version") != 2
                or not isinstance(row.get("domain_id"), str)
                or not row["domain_id"]
                or any(
                    not isinstance(row.get(field), str)
                    or not _SHA256.fullmatch(row[field])
                    for field in _IDENTITY_FIELDS[1:]
                )
                or row.get("provenance") != "future_trajectory_oracle"
                or type(stamp) is not int
                or not 0 < stamp <= 0xFFFFFFFF * 1_000_000_000 + 999_999_999
                or type(command) is not int
                or not 0 <= command < 4
                or row.get("command_name") != _COMMAND_NAMES[command]
                or type(row.get("timestamp_ns")) is not int
                or row["timestamp_ns"] <= 0
                or (
                    previous_timestamp is not None
                    and row["timestamp_ns"] <= previous_timestamp
                )
                or type(row.get("front_receipt_ns")) is not int
                or row["front_receipt_ns"] <= 0
                or type(row.get("front_record_index")) is not int
                or row["front_record_index"] < 0
                or (schedule and stamp <= schedule[-1][0])
                or any(
                    row.get(field) != expected_identity[field]
                    for field in _PROFILE_IDENTITY_FIELDS
                )
            ):
                raise ValueError("invalid command sidecar row")
            row_identity = tuple(row[field] for field in _IDENTITY_FIELDS)
            if identity is None:
                identity = row_identity
            elif row_identity != identity:
                raise ValueError("command sidecar identity changed")
            schedule.append((stamp, command))
            previous_timestamp = row["timestamp_ns"]
    if not schedule:
        raise ValueError("command sidecar is empty")
    return tuple(schedule)


def due_commands(schedule, next_index, clock_ns):
    """replay clock에 도달한 command row를 정확히 한 번 반환한다."""
    end = next_index
    while end < len(schedule) and schedule[end][0] <= clock_ns:
        end += 1
    return schedule[next_index:end], end


def select_output_stamp(source_stamp, use_ros_time_stamp):
    """명시적으로 요청한 경우에만 bag replay 시각을 선택한다."""
    return rospy.Time.now() if use_ros_time_stamp else source_stamp


def validate_command_replay(model_id, use_ros_time_stamp, reference_policy):
    """command sidecar에 필요한 atomic replay 조건을 검증한다."""
    if (
        model_id != "diffusiondrive_v2"
        or not use_ros_time_stamp
        or reference_policy != "new_stamp_once"
    ):
        raise ValueError(
            "command_sidecar requires diffusiondrive_v2, receipt stamps, "
            "and new_stamp_once"
        )


def _default_profile_file() -> str:
    """source/install package 위치에서 OptFeeder의 기본 profile.json을 찾는다.

    입력: 없음.
    반환: 존재하는 profile path 또는 우선 후보 path 문자열.
    """
    package_dir = Path(rospkg.RosPack().get_path("validator_inference"))
    candidates = (
        package_dir.parent / "config" / "profile.json",
        package_dir / "config" / "profile.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def _calibration_message(data: CameraCalibrationData) -> CameraCalibration:
    """profile에서 검증한 calibration 값 객체를 ROS message로 직렬화한다.

    입력: 한 camera의 ``CameraCalibrationData``.
    반환: ``DrivingFeature``에 넣을 ``CameraCalibration`` message.
    """
    message = CameraCalibration()
    message.name = data.name
    message.frame_id = data.frame_id
    message.image_width = data.image_width
    message.image_height = data.image_height
    message.k_original = list(data.k_original)
    message.distortion_model = data.distortion_model
    message.distortion_coefficients = list(data.distortion_coefficients)
    message.t_lidar_from_camera = list(data.t_lidar_from_camera)
    return message


class OptFeederNode:
    """단일 ego 구독에서 모든 model auxiliary feature를 발행하는 ROS1 node다.

    입력: ``/Ego_topic``과 선택 profile의 static calibration.
    반환: ``/driving_feature`` 및 필요한 fallback target topic.
    """

    def __init__(self) -> None:
        """profile별 publisher와 하나의 ``/Ego_topic`` subscription을 만든다.

        입력: ``~model_id``와 선택적 ``~profile_file`` ROS params.
        부작용: driving feature/target publisher와 ego subscriber를 등록한다.
        """
        self._model_id = rospy.get_param("~model_id")
        profile_file = rospy.get_param("~profile_file", _default_profile_file())
        self._profile = load_selected_profile(profile_file, self._model_id)
        self._use_ros_time_stamp = bool(
            rospy.get_param("~use_ros_time_stamp", False)
        )
        sidecar = rospy.get_param("~command_sidecar", "")
        if sidecar:
            validate_command_replay(
                self._model_id,
                self._use_ros_time_stamp,
                self._profile.data.get("inference_policy", {}).get(
                    "reference_stamp_policy"
                ),
            )
        profile_identity = self._profile.data.get("export_identity", {})
        expected_identity = {
            field: profile_identity.get(field)
            for field in _PROFILE_IDENTITY_FIELDS
        }
        self._command_schedule = (
            load_command_sidecar(sidecar, expected_identity) if sidecar else ()
        )
        self._ego_history = deque(maxlen=200)
        self._next_command = 0
        self._sidecar_lock = Lock()

        self._feature_publisher = rospy.Publisher(
            "/driving_feature", DrivingFeature, queue_size=1
        )
        self._target_data = build_fallback_targets(self._model_id)
        self._target_publishers = {
            target.topic: rospy.Publisher(target.topic, PointStamped, queue_size=1)
            for target in self._target_data
        }
        self._ego_subscriber = rospy.Subscriber(
            "/Ego_topic", EgoVehicleStatus, self._on_ego, queue_size=1
        )
        self._clock_subscriber = (
            rospy.Subscriber("/clock", Clock, self._on_clock, queue_size=1)
            if self._command_schedule
            else None
        )

    def _on_ego(self, ego: EgoVehicleStatus) -> None:
        """ego motion 한 샘플을 model-native feature와 fallback target으로 발행한다.

        입력: velocity·acceleration·header를 가진 ``EgoVehicleStatus``.
        부작용: 동기화된 stamp로 ``/driving_feature``와 route target을 발행한다.
        """
        if self._command_schedule:
            stamp = rospy.Time.now()
            with self._sidecar_lock:
                self._ego_history.append((stamp.to_nsec(), ego))
                self._drain(stamp)
            return
        self._publish(
            ego,
            select_output_stamp(ego.header.stamp, self._use_ros_time_stamp),
            1,
            "fallback",
        )

    def _on_clock(self, message: Clock) -> None:
        """rosbag replay 시각에 도달한 sidecar row를 발행한다."""
        with self._sidecar_lock:
            self._drain(message.clock)

    def _drain(self, clock_stamp) -> None:
        """도달한 oracle command를 receipt-grid stamp로 한 번씩 발행한다."""
        due, _ = due_commands(
            self._command_schedule,
            self._next_command,
            clock_stamp.to_nsec(),
        )
        for stamp_ns, command_index in due:
            if not self._ego_history or self._ego_history[-1][0] <= stamp_ns:
                return
            ego = next(
                (
                    candidate
                    for ego_stamp, candidate in reversed(self._ego_history)
                    if ego_stamp <= stamp_ns
                ),
                None,
            )
            if ego is None:
                return
            self._publish(
                ego,
                rospy.Time(
                    stamp_ns // 1_000_000_000,
                    stamp_ns % 1_000_000_000,
                ),
                command_index,
                "future_trajectory_oracle",
            )
            self._next_command += 1

    def _publish(self, ego, stamp, command_index, source) -> None:
        """model-native driving feature 하나를 직렬화해 발행한다."""
        data = build_driving_feature_data(
            self._model_id,
            vx=ego.velocity.x,
            vy=ego.velocity.y,
            ax=ego.acceleration.x,
            ay=ego.acceleration.y,
            command_index=command_index,
            source=source,
            profile_data=self._profile.data,
        )
        message = DrivingFeature()
        message.header.stamp = stamp
        message.header.frame_id = ego.header.frame_id
        message.model_id = data.model_id
        message.status_feature = list(data.status_feature)
        message.tfv6_radar = list(data.tfv6_radar)
        message.tfv6_radar_rows = data.tfv6_radar_rows
        message.tfv6_radar_cols = data.tfv6_radar_cols
        message.camera_calibrations = [
            _calibration_message(item) for item in data.camera_calibrations
        ]
        message.source = data.source
        self._feature_publisher.publish(message)

        for target in self._target_data:
            point = PointStamped()
            point.header.stamp = stamp
            point.header.frame_id = target.frame_id
            point.point.x = target.x
            point.point.y = target.y
            point.point.z = target.z
            self._target_publishers[target.topic].publish(point)


def main() -> None:
    """OptFeederNode를 초기화하고 ROS shutdown까지 spin한다.

    입력: 없음.
    부작용: ``OptFeederNode``라는 ROS node를 시작한다.
    """
    rospy.init_node("OptFeederNode")
    OptFeederNode()
    rospy.spin()


if __name__ == "__main__":
    main()
