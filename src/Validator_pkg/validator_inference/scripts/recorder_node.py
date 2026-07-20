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
    """open-loop inference outcome을 기록하는 유일한 CSV writer 노드."""

    def __init__(self) -> None:
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
        self._writer.append_metric(metric)


def main() -> None:
    rospy.init_node("RecorderNode")
    RecorderNode()
    rospy.spin()


if __name__ == "__main__":
    main()
