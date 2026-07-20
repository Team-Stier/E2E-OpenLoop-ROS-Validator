import pytest

from validator_inference.opt_features import (
    FeatureContractError,
    build_driving_feature_data,
    build_fallback_targets,
)


def _camera(name, width, height, seed):
    return {
        "name": name,
        "frame_id": f"Camera-{seed}",
        "image_size_px": [width, height],
        "K_original_3x3": [float(seed + index) for index in range(9)],
        "distortion_model": "plumb_bob",
        "distortion_coefficients": [0.1 * seed, 0.2 * seed],
        "T_lidar_from_camera_4x4": [float(seed + index) for index in range(16)],
    }


def _sparse_profile(cameras):
    return {
        "static_runtime_inputs": {
            "camera_calibration": {
                "source": "profile_inline",
                "cameras": cameras,
            }
        }
    }


def test_tfv6_builds_lanefollow_status_and_source_padded_empty_radar():
    feature = build_driving_feature_data("tfv6", vx=7.25)

    assert feature.status_feature == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 7.25)
    assert feature.tfv6_radar_rows == 300
    assert feature.tfv6_radar_cols == 5
    assert len(feature.tfv6_radar) == 1500
    assert feature.tfv6_radar[0:5] == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert feature.tfv6_radar[74 * 5 : 75 * 5] == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert feature.tfv6_radar[75 * 5 : 76 * 5] == (0.0, 0.0, 0.0, 0.0, 1.0)
    assert feature.tfv6_radar[225 * 5 : 226 * 5] == (0.0, 0.0, 0.0, 0.0, 3.0)
    assert feature.source == "fallback"


def test_interfuser_builds_lanefollow_status_without_radar():
    feature = build_driving_feature_data("interfuser", vx=4.5)

    assert feature.status_feature == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.5)
    assert feature.tfv6_radar == ()
    assert feature.tfv6_radar_rows == 0
    assert feature.tfv6_radar_cols == 0


@pytest.mark.parametrize("model_id", ["diffusiondrive_v2", "sparsedrive_v2"])
def test_navsim_models_preserve_command_velocity_and_acceleration_order(model_id):
    profile = _sparse_profile(
        [
            _camera("front", 1280, 720, 1),
            _camera("left", 640, 480, 2),
            _camera("right", 640, 480, 3),
        ]
    )
    if model_id != "sparsedrive_v2":
        profile = None

    feature = build_driving_feature_data(
        model_id, vx=1.0, vy=-2.0, ax=3.0, ay=-4.0, profile_data=profile
    )

    assert feature.status_feature == (0.0, 1.0, 0.0, 0.0, 1.0, -2.0, 3.0, -4.0)
    assert feature.tfv6_radar == ()


def test_sparse_calibration_is_converted_in_front_left_right_order():
    cameras = [
        _camera("right", 640, 480, 3),
        _camera("front", 1280, 720, 1),
        _camera("left", 640, 480, 2),
    ]

    feature = build_driving_feature_data(
        "sparsedrive_v2", vx=0.0, profile_data=_sparse_profile(cameras)
    )

    assert [camera.name for camera in feature.camera_calibrations] == [
        "front",
        "left",
        "right",
    ]
    front = feature.camera_calibrations[0]
    assert front.frame_id == "Camera-1"
    assert (front.image_width, front.image_height) == (1280, 720)
    assert front.k_original == tuple(cameras[1]["K_original_3x3"])
    assert front.distortion_model == "plumb_bob"
    assert front.distortion_coefficients == tuple(
        cameras[1]["distortion_coefficients"]
    )
    assert front.t_lidar_from_camera == tuple(
        cameras[1]["T_lidar_from_camera_4x4"]
    )


def test_sparse_malformed_inline_calibration_is_rejected_clearly():
    cameras = [
        _camera("front", 1280, 720, 1),
        _camera("left", 640, 480, 2),
        _camera("right", 640, 480, 3),
    ]
    cameras[1]["distortion_coefficients"] = []

    with pytest.raises(FeatureContractError, match="left.*distortion_coefficients"):
        build_driving_feature_data(
            "sparsedrive_v2", vx=0.0, profile_data=_sparse_profile(cameras)
        )


def test_tfv6_fallback_targets_use_only_required_topics_in_lidar_frame():
    targets = build_fallback_targets("tfv6")

    assert [(target.topic, target.x) for target in targets] == [
        ("/route/target_point_previous", 5.0),
        ("/route/target_point", 15.0),
        ("/route/target_point_next", 30.0),
    ]
    assert all(target.frame_id == "Lidar3D-4" for target in targets)
    assert all((target.y, target.z) == (0.0, 0.0) for target in targets)


@pytest.mark.parametrize(
    "model_id, expected_topics",
    [
        ("interfuser", ["/route/target_point"]),
        ("diffusiondrive_v2", []),
        ("sparsedrive_v2", []),
    ],
)
def test_models_publish_only_their_required_fallback_topics(
    model_id, expected_topics
):
    assert [target.topic for target in build_fallback_targets(model_id)] == expected_topics


def test_unknown_model_is_rejected():
    with pytest.raises(FeatureContractError, match="unsupported model_id"):
        build_driving_feature_data("unknown", vx=0.0)
