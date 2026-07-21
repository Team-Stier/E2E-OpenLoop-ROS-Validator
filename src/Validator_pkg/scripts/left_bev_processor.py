#!/usr/bin/env python3

import cv2
import numpy as np
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class ImageBEVProcessor:

    def __init__(self):

        self.bridge = CvBridge()
        self.latest_image = None

        # Left 카메라 BEV 고정 좌표
        # 순서: top-left, top-right, bottom-right, bottom-left
        self.points = [
            (463, 237),
            (590, 295),
            (294, 467),
            (203, 272)
        ]

        self.display_width = 960
        self.display_height = 540

        self.bev_width = 600
        self.bev_height = 800

        self.front_window_name = "Left Processed Image"
        self.bev_window_name = "Left BEV"

        cv2.namedWindow(
            self.front_window_name,
            cv2.WINDOW_AUTOSIZE
        )

        cv2.setMouseCallback(
            self.front_window_name,
            self.mouse_callback
        )

        self.subscriber = rospy.Subscriber(
            "/image/left/processed",
            Image,
            self.image_callback,
            queue_size=1
        )

        self.bev_publisher = rospy.Publisher(
            "/image/left/BEV",
            Image,
            queue_size=1
        )

        self.gui_timer = rospy.Timer(
            rospy.Duration(1.0 / 30.0),
            self.update_gui
        )

        rospy.loginfo("Left Image BEV Processor started")
        rospy.loginfo("Waiting for /image/left/processed")
        rospy.loginfo("Publishing Left BEV to /image/left/BEV")
        rospy.loginfo("Click points in this order:")
        rospy.loginfo("1: top-left")
        rospy.loginfo("2: top-right")
        rospy.loginfo("3: bottom-right")
        rospy.loginfo("4: bottom-left")
        rospy.loginfo("Press 'r' to reset selected points")

    def image_callback(self, msg):

        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except CvBridgeError as error:
            rospy.logerr(
                f"Image conversion failed: {error}"
            )
            return

        height, width = self.latest_image.shape[:2]

        rospy.loginfo_throttle(
            1.0,
            f"Left image received: {width}x{height}"
        )

    def mouse_callback(self, event, x, y, flags, param):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.latest_image is None:
            rospy.logwarn("No image has been received yet")
            return

        if len(self.points) >= 4:
            rospy.logwarn(
                "Four points are already selected. "
                "Press 'r' to reset."
            )
            return

        original_height, original_width = self.latest_image.shape[:2]

        original_x = int(
            x * original_width / self.display_width
        )
        original_y = int(
            y * original_height / self.display_height
        )

        self.points.append(
            (original_x, original_y)
        )

        rospy.loginfo(
            f"Point {len(self.points)}: "
            f"display=({x}, {y}), "
            f"original=({original_x}, {original_y})"
        )

        if len(self.points) == 4:
            rospy.loginfo("Four points selected")
            rospy.loginfo("Perspective transform ready")

    def create_bev(self, image):

        if len(self.points) != 4:
            return None

        source_points = np.float32(
            self.points
        )

        destination_points = np.float32([
            [0, 0],
            [self.bev_width - 1, 0],
            [self.bev_width - 1, self.bev_height - 1],
            [0, self.bev_height - 1]
        ])

        transform_matrix = cv2.getPerspectiveTransform(
            source_points,
            destination_points
        )

        bev_image = cv2.warpPerspective(
            image,
            transform_matrix,
            (self.bev_width, self.bev_height)
        )

        return bev_image

    def publish_bev(self, bev_image):

        try:
            bev_msg = self.bridge.cv2_to_imgmsg(
                bev_image,
                encoding="bgr8"
            )

            bev_msg.header.stamp = rospy.Time.now()
            bev_msg.header.frame_id = "left_bev"

            self.bev_publisher.publish(bev_msg)

            rospy.loginfo_throttle(
                2.0,
                f"Left BEV published: "
                f"{self.bev_width}x{self.bev_height}"
            )

        except CvBridgeError as error:
            rospy.logerr_throttle(
                2.0,
                f"BEV publish failed: {error}"
            )

    def update_gui(self, event):

        if self.latest_image is None:
            cv2.waitKey(1)
            return

        marked_image = self.latest_image.copy()

        for index, point in enumerate(self.points):

            cv2.circle(
                marked_image,
                point,
                8,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                marked_image,
                str(index + 1),
                (point[0] + 12, point[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        if len(self.points) == 4:

            polygon_points = np.array(
                self.points,
                dtype=np.int32
            )

            cv2.polylines(
                marked_image,
                [polygon_points],
                isClosed=True,
                color=(0, 255, 0),
                thickness=3
            )

        display_image = cv2.resize(
            marked_image,
            (self.display_width, self.display_height)
        )

        cv2.imshow(
            self.front_window_name,
            display_image
        )

        bev_image = self.create_bev(
            self.latest_image
        )

        if bev_image is not None:

            cv2.imshow(
                self.bev_window_name,
                bev_image
            )

            self.publish_bev(
                bev_image
            )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            self.points = [
                (463, 237),
                (590, 295),
                (294, 467),
                (203, 272)
            ]

            rospy.loginfo("Left BEV points restored")

            try:
                cv2.destroyWindow(
                    self.bev_window_name
                )
            except cv2.error:
                pass


def main():

    rospy.init_node("left_bev_processor")

    ImageBEVProcessor()

    try:
        rospy.spin()

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
