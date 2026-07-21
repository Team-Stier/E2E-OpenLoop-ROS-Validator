#!/usr/bin/env python3

"""
Camera BEV와 LiDAR BEV를 시간 동기화하여 하나의 디버깅 영상으로 합성한다.
선택된 네 모델의 Lidar3D-4 waypoint를 base_link BEV에 변환해 표시한다.

pub:
    /visualization/bev_overlay
        sensor_msgs/Image, bgr8
        Camera BEV 위에 LiDAR BEV와 waypoint를 합성한 영상

    /visualization/bev_triptych
        sensor_msgs/Image, bgr8
        Camera BEV, LiDAR BEV, Overlay를 좌우로 배치한 비교 영상

sub:
    /image/BEV
        sensor_msgs/Image, bgr8

    /lidar3D/BEV
        sensor_msgs/Image, mono8

    /waypoints
        nav_msgs/Path
        Lidar3D-4 기준 모델 예측 경로
"""

from collections import deque
import json
from pathlib import Path as FilePath
import threading
from typing import Deque, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rospkg
import rospy

from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Path
from sensor_msgs.msg import Image


def _stamp_to_nsec(stamp) -> int:
    """ROS1/ROS2 호환 stamp를 정수 nanosecond로 변환한다."""
    to_nsec = getattr(stamp, "to_nsec", None)
    if callable(to_nsec):
        return int(to_nsec())
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    raise TypeError("stamp must expose ROS second/nanosecond fields")


def _quaternion_rotation_matrix(quaternion) -> np.ndarray:
    """xyzw quaternion을 검증해 3x3 회전 행렬로 변환한다."""
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("waypoint transform quaternion must contain four values")
    x, y, z, w = values
    norm = float(np.dot(values, values))
    if norm <= np.finfo(float).eps:
        raise ValueError("waypoint transform quaternion must be nonzero")
    scale = 2.0 / norm
    return np.asarray(
        [
            [1.0 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1.0 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1.0 - scale * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_point_xyz(point, translation, rotation) -> np.ndarray:
    """child-frame point를 parent-frame 좌표로 rigid transform한다."""
    point_array = np.asarray(point, dtype=np.float64)
    translation_array = np.asarray(translation, dtype=np.float64)
    if point_array.shape != (3,) or translation_array.shape != (3,):
        raise ValueError("waypoint point and translation must contain three values")
    return rotation @ point_array + translation_array


def _default_profile_file() -> str:
    """Locate the inference-owned profile in source or install space."""
    package_dir = FilePath(rospkg.RosPack().get_path("validator_inference"))
    candidates = (
        package_dir.parent / "config" / "profile.json",
        package_dir / "config" / "profile.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def _load_waypoint_transform(profile_file: str):
    """Read the common inference-to-BEV transform from the profile contract."""
    payload = json.loads(FilePath(profile_file).read_text(encoding="utf-8"))
    try:
        return payload["common_inference_policy"]["static_transforms"][
            "T_base_from_lidar"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "profile is missing common T_base_from_lidar"
        ) from error


class BevVisualizer:
    def __init__(self) -> None:
        self.bridge = CvBridge()

        # 공통 BEV geometry
        self.frame_id = rospy.get_param(
            "/bev_geometry/frame_id",
            "base_link",
        )
        self.width = int(
            rospy.get_param("/bev_geometry/width")
        )
        self.height = int(
            rospy.get_param("/bev_geometry/height")
        )
        self.x_min = float(
            rospy.get_param("/bev_geometry/x_min")
        )
        self.x_max = float(
            rospy.get_param("/bev_geometry/x_max")
        )
        self.y_min = float(
            rospy.get_param("/bev_geometry/y_min")
        )
        self.y_max = float(
            rospy.get_param("/bev_geometry/y_max")
        )
        self.resolution = float(
            rospy.get_param("/bev_geometry/resolution")
        )
        profile_file = str(rospy.get_param("~profile_file", "")).strip()
        if not profile_file:
            profile_file = _default_profile_file()
        waypoint_transform = _load_waypoint_transform(profile_file)
        self.waypoint_parent_frame = str(
            waypoint_transform["parent_frame_id"]
        ).lstrip("/")
        self.waypoint_frame_id = str(
            waypoint_transform["child_frame_id"]
        ).lstrip("/")
        self.waypoint_translation = np.asarray(
            waypoint_transform["translation_m"],
            dtype=np.float64,
        )
        self.waypoint_rotation = _quaternion_rotation_matrix(
            waypoint_transform["quaternion_xyzw"]
        )

        # Visualizer parameters
        self.camera_topic = rospy.get_param(
            "~camera_bev_topic",
            "/image/BEV",
        )
        self.lidar_topic = rospy.get_param(
            "~lidar_bev_topic",
            "/lidar3D/BEV",
        )
        self.waypoints_topic = rospy.get_param(
            "~waypoints_topic",
            "/waypoints",
        )

        self.overlay_topic = rospy.get_param(
            "~overlay_topic",
            "/visualization/bev_overlay",
        )
        self.triptych_topic = rospy.get_param(
            "~triptych_topic",
            "/visualization/bev_triptych",
        )

        self.lidar_threshold = int(
            rospy.get_param("~lidar_threshold", 1)
        )
        self.lidar_grid_value = int(
            rospy.get_param("~lidar_grid_value", 60)
        )
        self.lidar_alpha = float(
            rospy.get_param("~lidar_alpha", 0.7)
        )
        self.sync_slop = float(
            rospy.get_param("~sync_slop", 0.05)
        )
        self.sync_queue_size = int(
            rospy.get_param("~sync_queue_size", 20)
        )

        self.lidar_color = np.array(
            [
                int(rospy.get_param("~lidar_color_b", 0)),
                int(rospy.get_param("~lidar_color_g", 255)),
                int(rospy.get_param("~lidar_color_r", 0)),
            ],
            dtype=np.uint8,
        )

        self.draw_waypoints_enabled = bool(
            rospy.get_param("~draw_waypoints", True)
        )
        waypoint_history_size = int(
            rospy.get_param("~waypoint_history_size", 50)
        )
        self.waypoint_max_age = float(
            rospy.get_param("~waypoint_max_age", 0.5)
        )
        self.waypoint_radius = int(
            rospy.get_param("~waypoint_radius", 3)
        )
        self.waypoint_line_thickness = int(
            rospy.get_param("~waypoint_line_thickness", 2)
        )

        if waypoint_history_size <= 0:
            raise ValueError("~waypoint_history_size must be positive")
        if self.waypoint_max_age < 0.0:
            raise ValueError("~waypoint_max_age must be non-negative")
        self.path_history: Deque[Path] = deque(maxlen=waypoint_history_size)
        self.path_lock = threading.Lock()

        self.overlay_pub = rospy.Publisher(
            self.overlay_topic,
            Image,
            queue_size=1,
        )
        self.triptych_pub = rospy.Publisher(
            self.triptych_topic,
            Image,
            queue_size=1,
        )

        rospy.Subscriber(
            self.waypoints_topic,
            Path,
            self.waypoints_callback,
            queue_size=1,
        )

        camera_sub = message_filters.Subscriber(
            self.camera_topic,
            Image,
            queue_size=1,
        )
        lidar_sub = message_filters.Subscriber(
            self.lidar_topic,
            Image,
            queue_size=1,
        )

        self.synchronizer = (
            message_filters.ApproximateTimeSynchronizer(
                [camera_sub, lidar_sub],
                queue_size=self.sync_queue_size,
                slop=self.sync_slop,
                allow_headerless=False,
            )
        )
        self.synchronizer.registerCallback(
            self.synchronized_bev_callback
        )

        self.validate_geometry_config()

        rospy.loginfo(
            "BEV visualizer started: camera=%s, lidar=%s, output=%s",
            self.camera_topic,
            self.lidar_topic,
            self.overlay_topic,
        )

    def validate_geometry_config(self) -> None:
        expected_height = int(
            round((self.x_max - self.x_min) / self.resolution)
        )
        expected_width = int(
            round((self.y_max - self.y_min) / self.resolution)
        )

        if expected_height != self.height:
            raise ValueError(
                "BEV height 설정 불일치: "
                f"configured={self.height}, calculated={expected_height}"
            )

        if expected_width != self.width:
            raise ValueError(
                "BEV width 설정 불일치: "
                f"configured={self.width}, calculated={expected_width}"
            )

        if self.waypoint_parent_frame != self.frame_id.lstrip("/"):
            raise ValueError(
                "waypoint transform parent must match BEV frame: "
                f"parent={self.waypoint_parent_frame}, bev={self.frame_id}"
            )
        if self.waypoint_translation.shape != (3,):
            raise ValueError("waypoint transform translation must contain three values")

    def waypoints_callback(self, msg: Path) -> None:
        with self.path_lock:
            self.path_history.append(msg)

    def path_at(self, reference_stamp) -> Optional[Path]:
        """BEV 시각 이하의 가장 최신 causal path를 age 제한 안에서 고른다."""
        reference_ns = _stamp_to_nsec(reference_stamp)
        with self.path_lock:
            history = tuple(self.path_history)
        candidates = [
            path
            for path in history
            if _stamp_to_nsec(path.header.stamp) <= reference_ns
        ]
        if not candidates:
            return None
        selected = max(
            candidates,
            key=lambda path: _stamp_to_nsec(path.header.stamp),
        )
        age_seconds = (
            reference_ns - _stamp_to_nsec(selected.header.stamp)
        ) / 1_000_000_000.0
        if age_seconds > self.waypoint_max_age:
            return None
        return selected

    def metric_to_pixel(
        self,
        x: float,
        y: float,
    ) -> Tuple[int, int]:
        """
        base_link 좌표를 BEV 영상의 row, col로 변환한다.

        x: 차량 전방 [m]
        y: 차량 좌측 [m]
        """

        row = int(
            np.floor((self.x_max - x) / self.resolution)
        )
        col = int(
            np.floor((self.y_max - y) / self.resolution)
        )

        return row, col

    def validate_input_images(
        self,
        camera_bev: np.ndarray,
        lidar_bev: np.ndarray,
    ) -> bool:
        if camera_bev.ndim != 3 or camera_bev.shape[2] != 3:
            rospy.logerr_throttle(
                1.0,
                "Camera BEV는 HxWx3 영상이어야 합니다. shape=%s",
                str(camera_bev.shape),
            )
            return False

        if lidar_bev.ndim != 2:
            rospy.logerr_throttle(
                1.0,
                "LiDAR BEV는 mono8 HxW 영상이어야 합니다. shape=%s",
                str(lidar_bev.shape),
            )
            return False

        if camera_bev.shape[:2] != lidar_bev.shape[:2]:
            rospy.logerr_throttle(
                1.0,
                "Camera/LiDAR BEV 크기 불일치: camera=%s, lidar=%s. "
                "cv2.resize로 강제로 맞추지 말고 processor 설정을 수정하십시오.",
                str(camera_bev.shape[:2]),
                str(lidar_bev.shape[:2]),
            )
            return False

        received_height, received_width = camera_bev.shape[:2]

        if (
            received_width != self.width
            or received_height != self.height
        ):
            rospy.logerr_throttle(
                1.0,
                "BEV 설정 크기와 수신 영상 크기 불일치: "
                "received=%dx%d, configured=%dx%d",
                received_width,
                received_height,
                self.width,
                self.height,
            )
            return False

        return True

    def create_lidar_color_image(
        self,
        lidar_bev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        LiDAR mono8 영상으로부터 BGR 색상 영상과 유효 mask를 만든다.
        """
        lidar_mask = lidar_bev >= self.lidar_threshold
        if 0 <= self.lidar_grid_value <= 255:
            lidar_mask &= lidar_bev != self.lidar_grid_value

        lidar_color_image = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8,
        )
        lidar_color_image[lidar_mask] = self.lidar_color

        return lidar_color_image, lidar_mask

    def blend_bev(
        self,
        camera_bev: np.ndarray,
        lidar_color_image: np.ndarray,
        lidar_mask: np.ndarray,
    ) -> np.ndarray:
        """
        LiDAR point가 존재하는 셀에만 색상 합성을 수행한다.
        """
        overlay = camera_bev.copy()

        camera_pixels = camera_bev[lidar_mask].astype(np.float32)
        lidar_pixels = lidar_color_image[lidar_mask].astype(np.float32)

        blended_pixels = (
            (1.0 - self.lidar_alpha) * camera_pixels
            + self.lidar_alpha * lidar_pixels
        )

        overlay[lidar_mask] = np.clip(
            blended_pixels,
            0,
            255,
        ).astype(np.uint8)

        return overlay

    def draw_waypoints(
        self,
        image: np.ndarray,
        reference_stamp,
    ) -> np.ndarray:
        if not self.draw_waypoints_enabled:
            return image

        path = self.path_at(reference_stamp)
        if path is None:
            return image

        path_frame = str(path.header.frame_id).lstrip("/")
        bev_frame = self.frame_id.lstrip("/")

        if path_frame not in ("", bev_frame, self.waypoint_frame_id):
            rospy.logwarn_throttle(
                2.0,
                "Waypoint frame %s은 BEV frame %s 또는 계약 frame %s이 아닙니다.",
                path_frame,
                bev_frame,
                self.waypoint_frame_id,
            )
            return image

        pixels = []

        for pose_stamped in path.poses:
            position = pose_stamped.pose.position
            point = np.asarray(
                [float(position.x), float(position.y), float(position.z)],
                dtype=np.float64,
            )
            if path_frame == self.waypoint_frame_id:
                point = transform_point_xyz(
                    point,
                    self.waypoint_translation,
                    self.waypoint_rotation,
                )

            row, col = self.metric_to_pixel(float(point[0]), float(point[1]))

            if (
                0 <= row < self.height
                and 0 <= col < self.width
            ):
                pixels.append([col, row])

        if len(pixels) >= 2:
            point_array = np.asarray(
                pixels,
                dtype=np.int32,
            )

            cv2.polylines(
                image,
                [point_array],
                isClosed=False,
                color=(255, 0, 0),  # BGR: 파랑
                thickness=self.waypoint_line_thickness,
                lineType=cv2.LINE_AA,
            )

        for col, row in pixels:
            cv2.circle(
                image,
                center=(col, row),
                radius=self.waypoint_radius,
                color=(0, 255, 255),  # BGR: 노랑
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        return image

    @staticmethod
    def add_label(
        image: np.ndarray,
        label: str,
    ) -> np.ndarray:
        output = image.copy()

        cv2.rectangle(
            output,
            (0, 0),
            (190, 30),
            color=(0, 0, 0),
            thickness=-1,
        )

        cv2.putText(
            output,
            label,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return output

    def synchronized_bev_callback(
        self,
        camera_msg: Image,
        lidar_msg: Image,
    ) -> None:
        expected_frame = self.frame_id.lstrip("/")
        for label, message in (("camera", camera_msg), ("lidar", lidar_msg)):
            observed_frame = str(message.header.frame_id).lstrip("/")
            if observed_frame and observed_frame != expected_frame:
                rospy.logerr_throttle(
                    1.0,
                    "%s BEV frame mismatch: received=%s configured=%s",
                    label,
                    observed_frame,
                    expected_frame,
                )
                return

        try:
            camera_bev = self.bridge.imgmsg_to_cv2(
                camera_msg,
                desired_encoding="bgr8",
            )

            lidar_bev = self.bridge.imgmsg_to_cv2(
                lidar_msg,
                desired_encoding="mono8",
            )

        except CvBridgeError as error:
            rospy.logerr_throttle(
                1.0,
                "cv_bridge conversion failed: %s",
                str(error),
            )
            return

        if not self.validate_input_images(
            camera_bev,
            lidar_bev,
        ):
            return

        lidar_color_image, lidar_mask = (
            self.create_lidar_color_image(lidar_bev)
        )

        overlay = self.blend_bev(
            camera_bev,
            lidar_color_image,
            lidar_mask,
        )

        reference_stamp = min(
            (camera_msg.header.stamp, lidar_msg.header.stamp),
            key=_stamp_to_nsec,
        )
        overlay = self.draw_waypoints(overlay, reference_stamp)

        camera_panel = self.add_label(
            camera_bev,
            "Camera BEV",
        )

        lidar_panel = self.add_label(
            lidar_color_image,
            "LiDAR BEV",
        )

        overlay_panel = self.add_label(
            overlay,
            "Overlay",
        )

        triptych = cv2.hconcat(
            [
                camera_panel,
                lidar_panel,
                overlay_panel,
            ]
        )

        output_stamp = camera_msg.header.stamp

        overlay_msg = self.bridge.cv2_to_imgmsg(
            overlay,
            encoding="bgr8",
        )
        overlay_msg.header.stamp = output_stamp
        overlay_msg.header.frame_id = self.frame_id

        triptych_msg = self.bridge.cv2_to_imgmsg(
            triptych,
            encoding="bgr8",
        )
        triptych_msg.header.stamp = output_stamp
        triptych_msg.header.frame_id = self.frame_id

        self.overlay_pub.publish(overlay_msg)
        self.triptych_pub.publish(triptych_msg)


def main() -> None:
    rospy.init_node("bev_visualizer")

    try:
        BevVisualizer()
        rospy.spin()

    except (
        KeyError,
        OSError,
        ValueError,
        rospkg.ResourceNotFound,
        rospy.ROSException,
    ) as error:
        rospy.logfatal("BEV visualizer initialization failed: %s", error)
        raise


if __name__ == "__main__":
    main()
