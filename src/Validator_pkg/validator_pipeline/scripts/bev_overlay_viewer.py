#!/usr/bin/env python3
"""Overlay synchronized camera and LiDAR BEVs and report their delay.

sub:
    /image/BEV                    sensor_msgs/Image, bgr8
    /lidar3D/BEV                  sensor_msgs/Image, mono8

pub:
    /image/BEV_overlay            sensor_msgs/Image, bgr8
    /bev_overlay/header_delay_ms  std_msgs/Float32
    /bev_overlay/arrival_delay_ms std_msgs/Float32

The signed delays are camera minus LiDAR. A negative value means that the
camera stamp or message arrival preceded the corresponding LiDAR value.
This node is visualization-only and does not feed data back into preprocessing.
"""

import time
from collections import OrderedDict, deque

import cv2
import message_filters
import numpy as np
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class BEVOverlayViewer:
    """Synchronize, blend, and publish camera/LiDAR BEV diagnostics."""

    def __init__(self):
        self.bridge = CvBridge()
        self.camera_arrivals = OrderedDict()
        self.lidar_arrivals = OrderedDict()
        self.camera_count = 0
        self.lidar_count = 0
        self.pair_count = 0

        self.camera_topic = rospy.get_param("~camera_topic", "/image/BEV")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/lidar3D/BEV")
        self.output_topic = rospy.get_param(
            "~output_topic",
            "/image/BEV_overlay",
        )
        self.header_delay_topic = rospy.get_param(
            "~header_delay_topic",
            "/bev_overlay/header_delay_ms",
        )
        self.arrival_delay_topic = rospy.get_param(
            "~arrival_delay_topic",
            "/bev_overlay/arrival_delay_ms",
        )
        self.output_frame_id = rospy.get_param(
            "~frame_id",
            "base_link",
        )
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.20))
        self.sync_queue_size = int(rospy.get_param("~sync_queue_size", 30))
        self.lidar_alpha = float(rospy.get_param("~lidar_alpha", 0.70))
        self.grid_value = int(rospy.get_param("~grid_value", 60))
        statistics_window = int(rospy.get_param("~statistics_window", 10000))

        if self.sync_slop < 0.0:
            raise ValueError("~sync_slop must be non-negative")
        if self.sync_queue_size <= 0:
            raise ValueError("~sync_queue_size must be positive")
        if not 0.0 <= self.lidar_alpha <= 1.0:
            raise ValueError("~lidar_alpha must be in [0, 1]")
        if statistics_window <= 0:
            raise ValueError("~statistics_window must be positive")

        self.header_delays_ms = deque(maxlen=statistics_window)
        self.arrival_delays_ms = deque(maxlen=statistics_window)

        self.overlay_publisher = rospy.Publisher(
            self.output_topic,
            Image,
            queue_size=1,
        )
        self.header_delay_publisher = rospy.Publisher(
            self.header_delay_topic,
            Float32,
            queue_size=10,
        )
        self.arrival_delay_publisher = rospy.Publisher(
            self.arrival_delay_topic,
            Float32,
            queue_size=10,
        )

        self.camera_subscriber = message_filters.Subscriber(
            self.camera_topic,
            Image,
            queue_size=5,
            buff_size=2**24,
        )
        self.lidar_subscriber = message_filters.Subscriber(
            self.lidar_topic,
            Image,
            queue_size=5,
            buff_size=2**24,
        )
        self.camera_subscriber.registerCallback(self.record_camera_arrival)
        self.lidar_subscriber.registerCallback(self.record_lidar_arrival)

        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.camera_subscriber, self.lidar_subscriber],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.synchronizer.registerCallback(self.overlay_callback)
        rospy.on_shutdown(self.report_statistics)

        rospy.loginfo("BEV overlay viewer initialized")
        rospy.loginfo(
            "Overlay: %s + %s -> %s (slop=%.3f s, lidar_alpha=%.2f)",
            self.camera_topic,
            self.lidar_topic,
            self.output_topic,
            self.sync_slop,
            self.lidar_alpha,
        )
        rospy.loginfo(
            "Delay topics: %s, %s",
            self.header_delay_topic,
            self.arrival_delay_topic,
        )

    @staticmethod
    def stamp_key(message):
        return message.header.stamp.to_nsec()

    @staticmethod
    def remember_arrival(cache, key):
        cache[key] = time.monotonic()
        while len(cache) > 200:
            cache.popitem(last=False)

    def record_camera_arrival(self, message):
        self.camera_count += 1
        self.remember_arrival(self.camera_arrivals, self.stamp_key(message))

    def record_lidar_arrival(self, message):
        self.lidar_count += 1
        self.remember_arrival(self.lidar_arrivals, self.stamp_key(message))

    @staticmethod
    def add_status_text(image, header_delay_ms, arrival_delay_ms):
        cv2.rectangle(image, (8, 8), (332, 70), (0, 0, 0), -1)
        cv2.putText(
            image,
            "stamp C-L: {:+7.1f} ms".format(header_delay_ms),
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        arrival_text = (
            "arrival C-L: {:+7.1f} ms".format(arrival_delay_ms)
            if np.isfinite(arrival_delay_ms)
            else "arrival C-L: unavailable"
        )
        cv2.putText(
            image,
            arrival_text,
            (18, 59),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def overlay_callback(self, camera_message, lidar_message):
        try:
            camera = self.bridge.imgmsg_to_cv2(
                camera_message,
                desired_encoding="bgr8",
            )
            lidar = self.bridge.imgmsg_to_cv2(
                lidar_message,
                desired_encoding="mono8",
            )
        except CvBridgeError as error:
            rospy.logerr_throttle(2.0, "BEV conversion failed: %s", error)
            return

        if camera.shape[:2] != lidar.shape[:2]:
            rospy.logerr_throttle(
                2.0,
                "BEV shape mismatch: camera=%s lidar=%s",
                camera.shape[:2],
                lidar.shape[:2],
            )
            return

        camera_key = self.stamp_key(camera_message)
        lidar_key = self.stamp_key(lidar_message)
        header_delay_ms = (
            camera_message.header.stamp - lidar_message.header.stamp
        ).to_sec() * 1000.0

        camera_arrival = self.camera_arrivals.pop(camera_key, None)
        lidar_arrival = self.lidar_arrivals.pop(lidar_key, None)
        if camera_arrival is None or lidar_arrival is None:
            arrival_delay_ms = float("nan")
        else:
            arrival_delay_ms = (camera_arrival - lidar_arrival) * 1000.0
            self.arrival_delays_ms.append(arrival_delay_ms)

        self.pair_count += 1
        self.header_delays_ms.append(header_delay_ms)

        lidar_color = cv2.applyColorMap(lidar, cv2.COLORMAP_TURBO)
        point_mask = lidar > 0
        if 0 <= self.grid_value <= 255:
            point_mask &= lidar != self.grid_value

        overlay = camera.copy()
        overlay[point_mask] = cv2.addWeighted(
            camera[point_mask],
            1.0 - self.lidar_alpha,
            lidar_color[point_mask],
            self.lidar_alpha,
            0.0,
        )
        self.add_status_text(overlay, header_delay_ms, arrival_delay_ms)

        output_message = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        output_message.header.seq = camera_message.header.seq
        output_message.header.stamp = camera_message.header.stamp
        output_message.header.frame_id = self.output_frame_id
        self.overlay_publisher.publish(output_message)
        self.header_delay_publisher.publish(Float32(data=header_delay_ms))
        if np.isfinite(arrival_delay_ms):
            self.arrival_delay_publisher.publish(Float32(data=arrival_delay_ms))

        if self.pair_count % 30 == 0:
            rospy.loginfo(
                "BEV pairs=%d, latest stamp C-L=%+.1f ms, "
                "arrival C-L=%+.1f ms",
                self.pair_count,
                header_delay_ms,
                arrival_delay_ms,
            )

    @staticmethod
    def summarize(label, values):
        if not values:
            rospy.loginfo("%s: no samples", label)
            return
        data = np.asarray(values, dtype=np.float64)
        absolute = np.abs(data)
        rospy.loginfo(
            "%s: samples=%d, signed_mean=%+.2f ms, "
            "abs_mean=%.2f ms, p95=%.2f ms, max=%.2f ms",
            label,
            data.size,
            float(data.mean()),
            float(absolute.mean()),
            float(np.percentile(absolute, 95)),
            float(absolute.max()),
        )

    def report_statistics(self):
        rospy.loginfo(
            "BEV overlay counts: camera=%d, lidar=%d, paired=%d",
            self.camera_count,
            self.lidar_count,
            self.pair_count,
        )
        self.summarize("Header stamp C-L", self.header_delays_ms)
        self.summarize("ROS arrival C-L", self.arrival_delays_ms)


if __name__ == "__main__":
    rospy.init_node("bev_overlay_viewer", anonymous=False)
    try:
        BEVOverlayViewer()
    except (ValueError, TypeError) as error:
        rospy.logfatal("BEV overlay viewer initialization failed: %s", error)
        raise SystemExit(1)
    rospy.spin()
