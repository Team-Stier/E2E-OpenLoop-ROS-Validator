#!/usr/bin/env python3
"""Project three adaptor outputs onto one vehicle-frame ground-plane BEV.

sub:
    /image/front/processed
    /image/left/processed
    /image/right/processed

pub:
    /image/BEV
    /image/bev/front
    /image/bev/left
    /image/bev/right
"""

import json
import math
import os

import cv2
import message_filters
import numpy as np
import rospkg
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


CAMERA_NAMES = ("front", "left", "right")


def load_camera_calibrations(config_path):
    """Load the three camera poses and intrinsics from a MORAI sensor JSON."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        sensor_config = json.load(config_file)

    calibrations = {}

    for camera in sensor_config.get("cameraList", []):
        camera_config = camera.get("cc", {})
        ros_topic = camera_config.get("rosConfig", {}).get("Topic", "").lower()
        camera_name = next(
            (name for name in CAMERA_NAMES if name in ros_topic),
            None,
        )

        if camera_name is None:
            continue

        position = camera["pos"]
        rotation = camera["rot"]
        principal_point = camera_config["principalPoint"]
        lens_distortion = camera_config.get(
            "lensDistortion",
            [0.0, 0.0, 0.0],
        )

        calibrations[camera_name] = {
            "position": np.array(
                [
                    float(position["x"]),
                    float(position["y"]),
                    float(position["z"]),
                ],
                dtype=np.float32,
            ),
            "roll": float(rotation["roll"]),
            "pitch": float(rotation["pitch"]),
            "yaw": float(rotation["yaw"]),
            "config_width": int(camera_config["cameraResWidth"]),
            "config_height": int(camera_config["cameraResHeight"]),
            "horizontal_fov": float(camera_config["cameraFOV"]),
            "principal_x": float(principal_point["x"]),
            "principal_y": float(principal_point["y"]),
            "radial_distortion": [
                float(value)
                for value in lens_distortion[:3]
            ],
        }

    missing = [name for name in CAMERA_NAMES if name not in calibrations]
    if missing:
        raise ValueError(
            "Missing camera calibration(s) in {}: {}".format(
                config_path,
                ", ".join(missing),
            )
        )

    return calibrations


def camera_axes(calibration):
    """Return camera right/down/forward axes in the MORAI vehicle frame."""
    roll = math.radians(calibration["roll"])
    pitch = math.radians(calibration["pitch"])
    yaw = math.radians(calibration["yaw"])

    # MORAI vehicle frame: x forward, y left, z up. Positive camera pitch in
    # the supplied configuration points the optical axis down toward the road.
    forward = np.array(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
        ],
        dtype=np.float32,
    )
    right = np.array(
        [math.sin(yaw), -math.cos(yaw), 0.0],
        dtype=np.float32,
    )
    down = np.cross(forward, right)

    if roll != 0.0:
        original_right = right.copy()
        original_down = down.copy()
        right = math.cos(roll) * original_right + math.sin(roll) * original_down
        down = -math.sin(roll) * original_right + math.cos(roll) * original_down

    return right, down, forward


class CameraBEVStitcher:
    """Synchronize adaptor outputs and publish a blended common-frame BEV."""

    def __init__(self):
        self.bridge = CvBridge()

        package_path = rospkg.RosPack().get_path("Validator_pkg")
        default_config_path = os.path.join(
            package_path,
            "config",
            "0716demo.json",
        )
        self.config_path = rospy.get_param(
            "~config_path",
            default_config_path,
        )
        self.calibrations = load_camera_calibrations(self.config_path)

        self.forward_range = float(rospy.get_param("~forward_range", 25.0))
        self.lateral_range = float(rospy.get_param("~lateral_range", 30.0))
        self.resolution = float(rospy.get_param("~resolution", 0.05))
        self.feather_pixels = float(rospy.get_param("~feather_pixels", 30.0))
        self.blend_exponent = float(rospy.get_param("~blend_exponent", 64.0))
        self.frame_id = rospy.get_param("~frame_id", "base_link")
        self.show_gui = bool(rospy.get_param("~show_gui", True))
        self.publish_debug_bevs = bool(
            rospy.get_param("~publish_debug_bevs", True)
        )

        for camera_name in CAMERA_NAMES:
            calibration = self.calibrations[camera_name]
            json_distortion = calibration["radial_distortion"]
            calibration["radial_k1"] = float(
                rospy.get_param(
                    "~{}_radial_k1".format(camera_name),
                    json_distortion[0],
                )
            )
            calibration["radial_k2"] = float(
                rospy.get_param(
                    "~{}_radial_k2".format(camera_name),
                    json_distortion[1],
                )
            )
            calibration["radial_k3"] = float(
                rospy.get_param(
                    "~{}_radial_k3".format(camera_name),
                    json_distortion[2],
                )
            )
            calibration["blend_priority"] = float(
                rospy.get_param(
                    "~{}_blend_priority".format(camera_name),
                    100.0 if camera_name == "front" else 1.0,
                )
            )

        if self.forward_range <= 0.0:
            raise ValueError("~forward_range must be positive")
        if self.lateral_range <= 0.0:
            raise ValueError("~lateral_range must be positive")
        if self.resolution <= 0.0:
            raise ValueError("~resolution must be positive")
        if self.feather_pixels <= 0.0:
            raise ValueError("~feather_pixels must be positive")
        if self.blend_exponent <= 0.0:
            raise ValueError("~blend_exponent must be positive")

        self.output_height = int(round(self.forward_range / self.resolution))
        self.output_width = int(round(self.lateral_range / self.resolution))
        self.ground_x, self.ground_y = self.create_ground_grid()
        self.projection_cache = {}

        front_topic = rospy.get_param(
            "~front_topic",
            "/image/front/processed",
        )
        left_topic = rospy.get_param(
            "~left_topic",
            "/image/left/processed",
        )
        right_topic = rospy.get_param(
            "~right_topic",
            "/image/right/processed",
        )
        output_topic = rospy.get_param(
            "~output_topic",
            "/image/BEV",
        )

        self.publisher = rospy.Publisher(
            output_topic,
            Image,
            queue_size=1,
        )
        self.debug_publishers = {}

        if self.publish_debug_bevs:
            self.debug_publishers = {
                camera_name: rospy.Publisher(
                    "/image/bev/{}".format(camera_name),
                    Image,
                    queue_size=1,
                )
                for camera_name in CAMERA_NAMES
            }

        self.subscribers = [
            message_filters.Subscriber(front_topic, Image, queue_size=1),
            message_filters.Subscriber(left_topic, Image, queue_size=1),
            message_filters.Subscriber(right_topic, Image, queue_size=1),
        ]
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            self.subscribers,
            queue_size=int(rospy.get_param("~sync_queue_size", 10)),
            slop=float(rospy.get_param("~sync_slop", 0.1)),
        )
        self.synchronizer.registerCallback(self.image_callback)

        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo("Camera BEV stitcher initialized")
        rospy.loginfo("Calibration: %s", self.config_path)
        rospy.loginfo(
            "BEV area: %.1f m forward x %.1f m lateral at %.2f m/pixel (%dx%d)",
            self.forward_range,
            self.lateral_range,
            self.resolution,
            self.output_width,
            self.output_height,
        )
        rospy.loginfo(
            "Input topics after image_adaptor: %s, %s, %s",
            front_topic,
            left_topic,
            right_topic,
        )
        rospy.loginfo("Publishing combined camera BEV to %s", output_topic)

        for camera_name in CAMERA_NAMES:
            calibration = self.calibrations[camera_name]
            position = calibration["position"]
            rospy.loginfo(
                "%s camera pose: xyz=(%.2f, %.2f, %.2f) m, "
                "rpy=(%.1f, %.1f, %.1f) deg, radial=(%.3f, %.3f, %.3f)",
                camera_name,
                position[0],
                position[1],
                position[2],
                calibration["roll"],
                calibration["pitch"],
                calibration["yaw"],
                calibration["radial_k1"],
                calibration["radial_k2"],
                calibration["radial_k3"],
            )

    def create_ground_grid(self):
        """Create vehicle-frame coordinates for each output BEV pixel."""
        pixel_x, pixel_y = np.meshgrid(
            np.arange(self.output_width, dtype=np.float32),
            np.arange(self.output_height, dtype=np.float32),
        )

        # Conventional vehicle BEV: forward is up, vehicle-left (+y) is on
        # the image left, and vehicle-right (-y) is on the image right.
        ground_x = self.forward_range - (pixel_y + 0.5) * self.resolution
        ground_y = (
            self.lateral_range / 2.0
            - (pixel_x + 0.5) * self.resolution
        )
        return ground_x, ground_y

    def get_projection(self, camera_name, image_shape):
        """Build or retrieve an output-to-source remap and blend weights."""
        image_height, image_width = image_shape[:2]
        cache_key = (camera_name, image_width, image_height)

        if cache_key in self.projection_cache:
            return self.projection_cache[cache_key]

        calibration = self.calibrations[camera_name]
        position = calibration["position"]
        right, down, forward = camera_axes(calibration)

        delta_x = self.ground_x - position[0]
        delta_y = self.ground_y - position[1]
        delta_z = -position[2]

        camera_x = right[0] * delta_x + right[1] * delta_y + right[2] * delta_z
        camera_y = down[0] * delta_x + down[1] * delta_y + down[2] * delta_z
        camera_z = forward[0] * delta_x + forward[1] * delta_y + forward[2] * delta_z

        horizontal_fov = math.radians(calibration["horizontal_fov"])
        focal_x = (image_width / 2.0) / math.tan(horizontal_fov / 2.0)
        focal_y = focal_x

        scale_x = image_width / float(calibration["config_width"])
        scale_y = image_height / float(calibration["config_height"])
        principal_x = calibration["principal_x"] * scale_x
        principal_y = calibration["principal_y"] * scale_y

        valid_depth = camera_z > 0.05
        normalized_x = np.zeros_like(camera_x)
        normalized_y = np.zeros_like(camera_y)
        np.divide(
            camera_x,
            camera_z,
            out=normalized_x,
            where=valid_depth,
        )
        np.divide(
            camera_y,
            camera_z,
            out=normalized_y,
            where=valid_depth,
        )
        radial_squared = normalized_x**2 + normalized_y**2
        radial_scale = (
            1.0
            + calibration["radial_k1"] * radial_squared
            + calibration["radial_k2"] * radial_squared**2
            + calibration["radial_k3"] * radial_squared**3
        )
        map_x = (
            focal_x * normalized_x * radial_scale + principal_x
        ).astype(np.float32)
        map_y = (
            focal_y * normalized_y * radial_scale + principal_y
        ).astype(np.float32)

        valid = (
            valid_depth
            & (map_x >= 0.0)
            & (map_x < image_width - 1)
            & (map_y >= 0.0)
            & (map_y < image_height - 1)
        )
        map_x[~valid] = -1.0
        map_y[~valid] = -1.0

        valid_mask = valid.astype(np.uint8)
        distance = cv2.distanceTransform(
            valid_mask,
            cv2.DIST_L2,
            3,
        )
        edge_weights = np.minimum(
            distance + valid_mask,
            self.feather_pixels,
        ).astype(np.float32) / self.feather_pixels
        center_cosine = np.clip(
            camera_z / np.maximum(
                np.sqrt(camera_x**2 + camera_z**2),
                1.0e-6,
            ),
            0.0,
            1.0,
        )
        view_weights = center_cosine**self.blend_exponent
        weights = (
            edge_weights
            * view_weights
            * calibration["blend_priority"]
        ).astype(np.float32)

        projection = (map_x, map_y, weights)
        self.projection_cache[cache_key] = projection
        return projection

    def image_callback(self, front_msg, left_msg, right_msg):
        messages = (front_msg, left_msg, right_msg)
        latest_message = max(
            messages,
            key=lambda message: message.header.stamp.to_nsec(),
        )

        try:
            images = [
                self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                for message in messages
            ]
        except CvBridgeError as error:
            rospy.logerr_throttle(2.0, "Camera image conversion failed: %s", error)
            return

        color_sum = np.zeros(
            (self.output_height, self.output_width, 3),
            dtype=np.float32,
        )
        weight_sum = np.zeros(
            (self.output_height, self.output_width),
            dtype=np.float32,
        )
        projected_images = {}

        for camera_name, image in zip(CAMERA_NAMES, images):
            map_x, map_y, weights = self.get_projection(
                camera_name,
                image.shape,
            )
            projected = cv2.remap(
                image,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            projected_images[camera_name] = projected
            color_sum += projected.astype(np.float32) * weights[:, :, None]
            weight_sum += weights

        if self.publish_debug_bevs:
            for camera_name, projected in projected_images.items():
                try:
                    debug_msg = self.bridge.cv2_to_imgmsg(
                        projected,
                        encoding="bgr8",
                    )
                    debug_msg.header.stamp = latest_message.header.stamp
                    debug_msg.header.frame_id = self.frame_id
                    self.debug_publishers[camera_name].publish(debug_msg)
                except CvBridgeError as error:
                    rospy.logerr_throttle(
                        2.0,
                        "%s debug BEV conversion failed: %s",
                        camera_name,
                        error,
                    )

        combined = np.zeros(
            (self.output_height, self.output_width, 3),
            dtype=np.uint8,
        )
        valid_output = weight_sum > 0.0

        if not np.any(valid_output):
            rospy.logwarn_throttle(2.0, "No camera projects onto the configured BEV area")
            return

        combined[valid_output] = np.clip(
            color_sum[valid_output] / weight_sum[valid_output, None],
            0.0,
            255.0,
        ).astype(np.uint8)

        try:
            output_msg = self.bridge.cv2_to_imgmsg(combined, encoding="bgr8")
        except CvBridgeError as error:
            rospy.logerr_throttle(2.0, "Combined BEV conversion failed: %s", error)
            return

        output_msg.header.stamp = latest_message.header.stamp
        output_msg.header.frame_id = self.frame_id
        self.publisher.publish(output_msg)

        if self.show_gui:
            try:
                cv2.imshow("Combined Camera BEV", combined)
                cv2.waitKey(1)
            except cv2.error as error:
                rospy.logwarn_throttle(5.0, "Combined BEV GUI unavailable: %s", error)

        rospy.loginfo_throttle(
            2.0,
            "Combined camera BEV published: %dx%d",
            self.output_width,
            self.output_height,
        )

    def on_shutdown(self):
        if self.show_gui:
            cv2.destroyAllWindows()


def main():
    rospy.init_node("camera_bev_stitcher")

    try:
        CameraBEVStitcher()
        rospy.spin()
    except (OSError, KeyError, TypeError, ValueError) as error:
        rospy.logfatal("Camera BEV stitcher initialization failed: %s", error)
        raise


if __name__ == "__main__":
    main()
