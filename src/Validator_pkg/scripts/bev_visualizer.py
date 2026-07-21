#!/usr/bin/env python3

"""
Camera BEV와 LiDAR BEV를 시간 동기화하여 하나의 디버깅 영상으로 합성한다.
선택적으로 TFv6가 출력한 waypoint를 동일 BEV 영상에 표시한다.

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
        base_link 기준 TFv6 예측 경로
"""

from typing import Optional, Tuple

import cv2
import message_filters
import numpy as np
import rospy

from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Path
from sensor_msgs.msg import Image


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
        self.waypoint_radius = int(
            rospy.get_param("~waypoint_radius", 3)
        )
        self.waypoint_line_thickness = int(
            rospy.get_param("~waypoint_line_thickness", 2)
        )

        self.latest_path: Optional[Path] = None

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

    def waypoints_callback(self, msg: Path) -> None:
        self.latest_path = msg

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
    ) -> np.ndarray:
        if not self.draw_waypoints_enabled:
            return image

        if self.latest_path is None:
            return image

        path_frame = self.latest_path.header.frame_id

        if path_frame not in ("", self.frame_id):
            rospy.logwarn_throttle(
                2.0,
                "Waypoint frame이 %s입니다. 현재 visualizer는 %s만 지원합니다. "
                "추론 노드에서 waypoint를 base_link 기준으로 발행하거나 "
                "tf2 변환을 추가하십시오.",
                path_frame,
                self.frame_id,
            )
            return image

        pixels = []

        for pose_stamped in self.latest_path.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y

            row, col = self.metric_to_pixel(x, y)

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

        overlay = self.draw_waypoints(overlay)

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

    except (KeyError, ValueError, rospy.ROSException) as error:
        rospy.logfatal("BEV visualizer initialization failed: %s", error)
        raise


if __name__ == "__main__":
    main()
