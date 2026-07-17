#!/usr/bin/env python3
"""
ROS bag의 압축 카메라 영상을 수신하고,
OpenCV 이미지로 디코딩한 뒤 sensor_msgs/Image 형식으로 다시 발행한다.

sub:
    /image_jpeg/front/compressed
    /image_jpeg/left/compressed
    /image_jpeg/right/compressed

pub:
    /image/front/processed
    /image/left/processed
    /image/right/processed
"""

import cv2
import numpy as np
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image


class ImageAdaptor:
    """MORAI 압축 카메라 영상을 일반 ROS Image로 변환하는 노드."""

    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.front_pub = rospy.Publisher(
            "/image/front/processed",
            Image,
            queue_size=1,
        )
        self.left_pub = rospy.Publisher(
            "/image/left/processed",
            Image,
            queue_size=1,
        )
        self.right_pub = rospy.Publisher(
            "/image/right/processed",
            Image,
            queue_size=1,
        )

        self.front_sub = rospy.Subscriber(
            "/image_jpeg/front/compressed",
            CompressedImage,
            self.front_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.left_sub = rospy.Subscriber(
            "/image_jpeg/left/compressed",
            CompressedImage,
            self.left_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.right_sub = rospy.Subscriber(
            "/image_jpeg/right/compressed",
            CompressedImage,
            self.right_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo("image_adaptor node initialized")

    def decode_compressed_image(self, msg: CompressedImage):
        """CompressedImage 메시지를 OpenCV BGR 이미지로 변환한다."""
        np_array = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            rospy.logwarn_throttle(
                2.0,
                "Failed to decode compressed image",
            )
            return None

        return image

    def publish_image(
        self,
        msg: CompressedImage,
        publisher: rospy.Publisher,
        camera_name: str,
    ) -> None:
        """이미지를 디코딩하고 원본 헤더를 유지한 채 발행한다."""
        image = self.decode_compressed_image(msg)

        if image is None:
            return

        try:
            output_msg = self.bridge.cv2_to_imgmsg(
                image,
                encoding="bgr8",
            )

            # 원본 rosbag 타임스탬프와 frame_id 유지
            output_msg.header = msg.header

            publisher.publish(output_msg)

            rospy.loginfo_throttle(
                5.0,
                "%s camera processed: %dx%d",
                camera_name,
                image.shape[1],
                image.shape[0],
            )

        except CvBridgeError as error:
            rospy.logerr_throttle(
                2.0,
                "CvBridge error on %s camera: %s",
                camera_name,
                str(error),
            )

    def front_callback(self, msg: CompressedImage) -> None:
        self.publish_image(msg, self.front_pub, "front")

    def left_callback(self, msg: CompressedImage) -> None:
        self.publish_image(msg, self.left_pub, "left")

    def right_callback(self, msg: CompressedImage) -> None:
        self.publish_image(msg, self.right_pub, "right")


def main() -> None:
    rospy.init_node("image_adaptor", anonymous=False)
    ImageAdaptor()
    rospy.spin()


if __name__ == "__main__":
    main()