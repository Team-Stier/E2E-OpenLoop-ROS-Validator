#!/usr/bin/env python3
"""
/Ego_topic을 모델별 non-perception feature로 변환하는 OptFeeder ROS1 노드.
입력: /Ego_topic, 로컬 profile.json.
출력(pub): /driving_feature, 필요한 /route/target_point*.
구독(sub): /Ego_topic 하나.
"""

from pathlib import Path

import rospkg
import rospy
from geometry_msgs.msg import PointStamped
from morai_msgs.msg import EgoVehicleStatus
from validator_msgs.msg import CameraCalibration, DrivingFeature

from validator_inference.opt_features import (
    CameraCalibrationData,
    build_driving_feature_data,
    build_fallback_targets,
)
from validator_inference.profile_store import load_selected_profile


def _default_profile_file() -> str:
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
    """단일 ego 구독과 선택 모델의 선택적 publisher를 소유한다."""

    def __init__(self) -> None:
        self._model_id = rospy.get_param("~model_id")
        profile_file = rospy.get_param("~profile_file", _default_profile_file())
        self._profile = load_selected_profile(profile_file, self._model_id)

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

    def _on_ego(self, ego: EgoVehicleStatus) -> None:
        data = build_driving_feature_data(
            self._model_id,
            vx=ego.velocity.x,
            vy=ego.velocity.y,
            ax=ego.acceleration.x,
            ay=ego.acceleration.y,
            profile_data=self._profile.data,
        )
        message = DrivingFeature()
        message.header.stamp = ego.header.stamp
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
            point.header.stamp = ego.header.stamp
            point.header.frame_id = target.frame_id
            point.point.x = target.x
            point.point.y = target.y
            point.point.z = target.z
            self._target_publishers[target.topic].publish(point)


def main() -> None:
    rospy.init_node("OptFeederNode")
    OptFeederNode()
    rospy.spin()


if __name__ == "__main__":
    main()
