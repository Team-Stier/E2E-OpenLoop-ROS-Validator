#!/usr/bin/env python3
"""ROS1 entrypoint for one selected open-loop inference runtime."""

from datetime import datetime, timezone
from pathlib import Path
import uuid

import rospkg
import rospy
from nav_msgs.msg import Path as RosPath
from validator_msgs.msg import InferenceMetric

from validator_inference.engine import InferenceEngine
from validator_inference.input_buffer import InputBuffer
from validator_inference.models.select import build_selected_model
from validator_inference.path_builder import PathMessageBuilder
from validator_inference.profile_store import load_selected_profile
from validator_inference.worker import InferenceWorker


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


def _new_run_id() -> str:
    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return "{}-{}".format(started, uuid.uuid4().hex[:8])


def require_sim_time(get_param) -> None:
    if get_param("/use_sim_time", False) is not True:
        raise RuntimeError(
            "InferenceEngineNode requires /use_sim_time=true for rosbag open-loop time"
        )


class InferenceEngineNode:
    def __init__(self) -> None:
        require_sim_time(rospy.get_param)
        model_id = rospy.get_param("~model_id")
        profile_file = rospy.get_param("~profile_file", _default_profile_file())
        run_id = rospy.get_param("~run_id", _new_run_id())

        selected_profile = load_selected_profile(profile_file, model_id)
        model = build_selected_model(selected_profile)
        model.load()

        input_buffer = InputBuffer.configure(selected_profile)
        worker = InferenceWorker(model, run_id)
        path_builder = PathMessageBuilder(
            selected_profile.data["static_transforms"]["T_base_from_lidar"]
        )
        path_publisher = rospy.Publisher("/waypoints", RosPath, queue_size=1)
        metric_publisher = rospy.Publisher(
            "/inference_metric", InferenceMetric, queue_size=10
        )
        self.engine = InferenceEngine(
            selected_profile=selected_profile,
            input_buffer=input_buffer,
            worker=worker,
            path_builder=path_builder,
            path_publisher=path_publisher,
            metric_publisher=metric_publisher,
            run_id=run_id,
        )
        self._timer = rospy.Timer(
            rospy.Duration.from_sec(0.1), self.engine.on_timer_tick
        )


def main() -> None:
    rospy.init_node("InferenceEngineNode")
    InferenceEngineNode()
    rospy.spin()


if __name__ == "__main__":
    main()
