#!/usr/bin/env python3
"""Subscribe to inference metrics and persist one CSV row per message."""

from pathlib import Path

import rospy
from validator_msgs.msg import InferenceMetric

from validator_inference.csv_recorder import CsvRunWriter, default_record_dir


class RecorderNode:
    """The sole CSV writer for open-loop inference outcomes."""

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
