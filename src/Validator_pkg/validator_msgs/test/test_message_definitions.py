from pathlib import Path


def test_message_contracts_are_compileable_ros1_fields():
    root = Path(__file__).parents[1] / "msg"
    camera = (root / "CameraCalibration.msg").read_text()
    driving = (root / "DrivingFeature.msg").read_text()
    metric = (root / "InferenceMetric.msg").read_text()

    assert "float64[16] t_lidar_from_camera" in camera
    assert "std_msgs/Header header" in driving
    assert "float32[] tfv6_radar" in driving
    assert "uint32 tfv6_radar_rows" in driving
    assert "uint32 tfv6_radar_cols" in driving
    assert "CameraCalibration[] camera_calibrations" in driving
    assert "string run_id" in metric
    assert "string[] input_keys" in metric
    assert "float64[] input_age_ms" in metric
    assert "string profile_sha256" in metric
