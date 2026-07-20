#!/usr/bin/env python3
"""
inference metric을 평가별 CSV 한 행으로 기록하는 Recorder ROS1 노드.
입력: /inference_metric.
출력(pub): 없음. records/openloop/ CSV에 기록.
구독(sub): /inference_metric 하나.
"""

from pathlib import Path

import rospy
from validator_msgs.msg import InferenceMetric

from validator_inference.csv_recorder import CsvRunWriter, default_record_dir


class RecorderNode:
    """open-loop inference metric을 평가당 하나의 CSV에 기록하는 유일한 writer다.

    입력: ``/inference_metric``과 record 경로 관련 ROS params.
    반환: 없음. 수신 순서대로 한 행씩 CSV를 갱신한다.
    """

    def __init__(self) -> None:
        """평가 CSV와 metric subscriber를 만들고 shutdown close handler를 등록한다.

        입력: ``~model_id``, 선택적 ``~record_dir``, ``~checkpoint_manifest`` params.
        부작용: 파일을 즉시 생성하고 ``/inference_metric``을 구독한다.
        """
        model_id = rospy.get_param("~model_id")
        record_dir = Path(
            rospy.get_param("~record_dir", str(default_record_dir()))
        )
        checkpoint_manifest = rospy.get_param("~checkpoint_manifest", "")
        self._writer = CsvRunWriter(
            record_dir,
            model_id,
            checkpoint_manifest=checkpoint_manifest,
        )
        rospy.on_shutdown(self._writer.close)
        self._metric_subscriber = rospy.Subscriber(
            "/inference_metric", InferenceMetric, self._on_metric, queue_size=100
        )

    def _on_metric(self, metric: InferenceMetric) -> None:
        """수신한 tick outcome 하나를 CSV writer에 위임한다.

        입력: engine이 발행한 ``InferenceMetric``.
        부작용: 현재 평가 CSV에 정확히 한 행을 append한다.
        """
        self._writer.append_metric(metric)


def main() -> None:
    """RecorderNode를 초기화하고 ROS shutdown까지 spin한다.

    입력: 없음.
    부작용: ``RecorderNode``라는 ROS node와 평가 CSV를 시작한다.
    """
    rospy.init_node("RecorderNode")
    RecorderNode()
    rospy.spin()


if __name__ == "__main__":
    main()
