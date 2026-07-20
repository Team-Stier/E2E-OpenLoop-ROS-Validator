#!/usr/bin/env python3
"""
선택된 모델의 open-loop inference engine ROS1 진입점.
입력: profile별 perception topic과 /driving_feature, 필요한 /route/target_point*, /tf.
출력(pub): /waypoints, /inference_metric.
구독(sub): InputBuffer가 위 입력 토픽을 단독 소유한다.
"""

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
    """source와 install 배치 모두에서 기본 profile.json 위치를 찾는다.

    입력: 없음.
    반환: 존재하는 config profile path, 없으면 우선 후보 path 문자열.
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


def _new_run_id() -> str:
    """CSV와 metric을 묶을 UTC 시각 기반의 충돌 가능성 낮은 run id를 만든다.

    입력: 없음.
    반환: UTC timestamp와 짧은 UUID suffix를 합친 run id.
    """
    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return "{}-{}".format(started, uuid.uuid4().hex[:8])


def require_sim_time(get_param) -> None:
    """rosbag open-loop 평가가 simulated clock을 사용하도록 강제한다.

    입력: ROS parameter getter.
    부작용: ``/use_sim_time``이 참이 아니면 ``RuntimeError``를 발생시킨다.
    """
    if get_param("/use_sim_time", False) is not True:
        raise RuntimeError(
            "InferenceEngineNode requires /use_sim_time=true for rosbag open-loop time"
        )


class InferenceEngineNode:
    """inference layer의 buffer·worker·engine을 조립하는 ROS1 process다.

    입력: ROS private params와 profile별 perception/feature topics.
    반환: ``/waypoints``와 ``/inference_metric``을 발행하는 10 Hz engine.
    """

    def __init__(self) -> None:
        """profile을 load하고 concrete model, buffer, worker, publisher, timer를 만든다.

        입력: ``~model_id``, 선택적 ``~profile_file``, 선택적 ``~run_id`` ROS params.
        부작용: model을 load하고 topic subscription 및 10 Hz timer를 시작한다.
        """
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
    """InferenceEngineNode를 초기화하고 ROS shutdown까지 spin한다.

    입력: 없음.
    부작용: ``InferenceEngineNode``라는 ROS node를 시작한다.
    """
    rospy.init_node("InferenceEngineNode")
    InferenceEngineNode()
    rospy.spin()


if __name__ == "__main__":
    main()
