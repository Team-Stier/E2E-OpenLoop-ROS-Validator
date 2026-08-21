"""Dynamic producer/visualizer contract checks with synthetic sensor data."""

from collections import deque
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPTS = PACKAGE_ROOT.parent / "validator_pipeline" / "scripts"


def _load_pipeline_script(name):
    pytest.importorskip("rospy")
    path = PIPELINE_SCRIPTS / name
    spec = spec_from_file_location(
        "validator_pipeline_test_{}".format(name.replace(".py", "")),
        path,
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def image_adaptor():
    return _load_pipeline_script("image_adaptor.py")


@pytest.fixture(scope="module")
def camera_stitcher():
    return _load_pipeline_script("camera_bev_stitcher.py")


@pytest.fixture(scope="module")
def lidar_processor():
    return _load_pipeline_script("lidar_bev_processor.py")


@pytest.fixture(scope="module")
def visualizer():
    return _load_pipeline_script("bev_visualizer.py")


@pytest.fixture(scope="module")
def camera_images():
    images = {
        "left": np.empty((480, 640, 3), dtype=np.uint8),
        "front": np.empty((720, 1280, 3), dtype=np.uint8),
        "right": np.empty((480, 640, 3), dtype=np.uint8),
    }
    images["left"][:] = (10, 20, 30)
    images["front"][:] = (40, 50, 60)
    images["right"][:] = (70, 80, 90)
    return images


def test_all_camera_profiles_emit_model_transport_contracts(
    image_adaptor,
    camera_images,
):
    tfv6 = image_adaptor.prepare_tfv6(camera_images)
    assert tfv6.shape == (384, 1152, 3)
    assert tfv6.dtype == np.uint8
    assert tfv6.flags.c_contiguous
    assert tuple(tfv6[0, 100]) == (30, 20, 10)
    assert tuple(tfv6[0, 500]) == (60, 50, 40)
    assert tuple(tfv6[0, 1000]) == (90, 80, 70)

    interfuser = image_adaptor.prepare_interfuser(camera_images)
    assert {name: value.shape for name, value in interfuser.items()} == {
        "front": (224, 224, 3),
        "left": (128, 128, 3),
        "right": (128, 128, 3),
        "center": (128, 128, 3),
    }
    assert all(value.dtype == np.float32 for value in interfuser.values())
    assert all(np.isfinite(value).all() for value in interfuser.values())

    diffusion = image_adaptor.prepare_diffusiondrive_v2(camera_images)
    assert diffusion.shape == (256, 1024, 3)
    assert diffusion.dtype == np.float32
    assert diffusion.flags.c_contiguous
    assert 0.0 <= float(diffusion.min()) <= float(diffusion.max()) <= 1.0
    np.testing.assert_allclose(
        diffusion[100, 100],
        np.asarray([30, 20, 10], dtype=np.float32) / 255.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        diffusion[100, 512],
        np.asarray([60, 50, 40], dtype=np.float32) / 255.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        diffusion[100, 900],
        np.asarray([90, 80, 70], dtype=np.float32) / 255.0,
        atol=1.0e-6,
    )

    sparse = image_adaptor.prepare_sparsedrive_v2(camera_images)
    assert set(sparse) == {"left", "front", "right"}
    assert all(value.shape == (256, 512, 3) for value in sparse.values())
    assert all(value.dtype == np.float32 for value in sparse.values())
    assert all(np.isfinite(value).all() for value in sparse.values())


def test_diffusiondrive_camera_matches_morai_training_adapter(image_adaptor):
    images = {
        "left": np.arange(480 * 640 * 3, dtype=np.uint32)
        .reshape(480, 640, 3)
        .astype(np.uint8),
        "front": np.arange(720 * 1280 * 3, dtype=np.uint32)
        .reshape(720, 1280, 3)
        .astype(np.uint8),
        "right": np.arange(480 * 640 * 3, dtype=np.uint32)[::-1]
        .reshape(480, 640, 3)
        .astype(np.uint8),
    }
    resized = [
        cv2.resize(
            cv2.cvtColor(images[name], cv2.COLOR_BGR2RGB),
            size,
            interpolation=cv2.INTER_AREA,
        )
        for name, size in (
            ("left", (341, 256)),
            ("front", (455, 256)),
            ("right", (341, 256)),
        )
    ]
    expected = (
        np.concatenate(resized, axis=1)[:, 56:1080]
        .astype(np.float32)
        / np.float32(255.0)
    )

    actual = image_adaptor.prepare_diffusiondrive_v2(
        images,
        morai_domain=True,
    )

    np.testing.assert_array_equal(actual, expected)

    canonical = {
        name: cv2.resize(
            cv2.cvtColor(images[name], cv2.COLOR_BGR2RGB),
            (1920, 1080),
            interpolation=cv2.INTER_LINEAR,
        )
        for name in image_adaptor.CAMERA_NAMES
    }
    official = cv2.resize(
        np.concatenate(
            [
                canonical["left"][28:-28, 416:-416],
                canonical["front"][28:-28],
                canonical["right"][28:-28, 416:-416],
            ],
            axis=1,
        ),
        (1024, 256),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32) / np.float32(255.0)
    np.testing.assert_array_equal(
        image_adaptor.prepare_diffusiondrive_v2(images),
        official,
    )


def test_morai_camera_rotations_match_recorded_optical_frame_transforms(
    image_adaptor,
    camera_stitcher,
):
    config_path = (
        PACKAGE_ROOT.parent
        / "validator_pipeline"
        / "config"
        / "0716demo.json"
    )
    sparse_calibration = image_adaptor.load_sparsedrive_calibration(config_path)
    bev_calibrations = camera_stitcher.load_camera_calibrations(config_path)
    expected_camera_to_lidar = {
        "front": np.asarray(
            [
                [0.0, -0.03489949670250097, 0.9993908270190958],
                [-1.0, 0.0, 0.0],
                [0.0, -0.9993908270190958, -0.03489949670250097],
            ]
        ),
        "left": np.asarray(
            [
                [0.9254165783983233, -0.17364817766693033, 0.33682408883346526],
                [-0.3420201433256688, 0.0, 0.9396926207859083],
                [-0.1631759111665348, -0.984807753012208, -0.05939117461388472],
            ]
        ),
        "right": np.asarray(
            [
                [-0.9254165783983233, -0.17364817766693033, 0.33682408883346543],
                [-0.342020143325669, 0.0, -0.9396926207859083],
                [0.1631759111665348, -0.984807753012208, -0.05939117461388475],
            ]
        ),
    }

    for camera_name, expected in expected_camera_to_lidar.items():
        vehicle_to_optical = (
            image_adaptor.camera_vehicle_to_optical_rotation(
                sparse_calibration["cameras"][camera_name]
            )
        )
        np.testing.assert_allclose(
            vehicle_to_optical,
            expected.T,
            atol=1.0e-12,
        )

        right, down, forward = camera_stitcher.camera_axes(
            bev_calibrations[camera_name]
        )
        np.testing.assert_allclose(
            np.stack([right, down, forward], axis=1),
            expected,
            atol=1.0e-7,
        )


def test_lidar_profiles_emit_expected_hwc_transport_shapes(lidar_processor):
    points = np.asarray(
        [
            [5.0, 0.0, 1.0],
            [10.0, 2.0, -1.0],
            [15.0, -3.0, 0.5],
            [2.0, 0.0, -2.0],
        ],
        dtype=np.float32,
    )

    visual = lidar_processor.prepare_visual_bev(points)
    assert visual.shape == (500, 600)
    assert visual.dtype == np.uint8
    assert visual.flags.c_contiguous

    tfv6 = lidar_processor.prepare_tfv6(points)
    assert tfv6.shape == (320, 384)
    assert tfv6.dtype == np.float32
    assert 0.0 <= float(tfv6.min()) <= float(tfv6.max()) <= 1.0

    interfuser = lidar_processor.prepare_interfuser(points)
    assert interfuser.shape == (224, 224, 3)
    assert interfuser.dtype == np.float32
    assert 0.0 <= float(interfuser[:, :, :2].min())
    assert float(interfuser[:, :, :2].max()) <= 1.0
    assert float(interfuser[:, :, 2].max()) <= 2.0

    diffusion = lidar_processor.prepare_diffusiondrive_v2(
        points,
        morai_domain=True,
    )
    assert diffusion.shape == (256, 256)
    assert diffusion.dtype == np.float32
    assert 0.0 <= float(diffusion.min()) <= float(diffusion.max()) <= 1.0


def test_diffusiondrive_lidar_removes_only_the_3d_vehicle_roof(lidar_processor):
    points = np.asarray(
        [
            [1.0, 0.0, 1.0],
            [0.0, -1.0, 0.8],
            [2.0, 1.0, 1.6],
            [1.0, 0.0, 0.5],
            [3.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    default = lidar_processor.prepare_diffusiondrive_v2(points)
    diffusion = lidar_processor.prepare_diffusiondrive_v2(
        points,
        morai_domain=True,
    )

    x_inside = int((1.0 + 32.0) * 4.0)
    x_ahead = int((3.0 + 32.0) * 4.0)
    y_center = int((0.0 + 32.0) * 4.0)
    assert default[x_inside, y_center] == pytest.approx(0.4)
    assert diffusion[x_inside, y_center] == pytest.approx(0.2)
    assert diffusion[x_ahead, y_center] == pytest.approx(0.2)
    assert diffusion[int(32.0 * 4.0), int(31.0 * 4.0)] == 0.0
    assert diffusion[int(34.0 * 4.0), int(33.0 * 4.0)] == 0.0


def test_replay_stamp_switch_uses_ros_clock(
    image_adaptor,
    lidar_processor,
    monkeypatch,
):
    source_stamp = object()
    replay_stamp = object()

    class FakeTime:
        @staticmethod
        def now():
            return replay_stamp

    for module in (image_adaptor, lidar_processor):
        monkeypatch.setattr(module.rospy, "Time", FakeTime)
        assert module.select_output_stamp(source_stamp, False) is source_stamp
        assert module.select_output_stamp(source_stamp, True) is replay_stamp

    class Bridge:
        @staticmethod
        def cv2_to_imgmsg(_image, encoding):
            return SimpleNamespace(
                encoding=encoding,
                header=SimpleNamespace(seq=0, stamp=None, frame_id=""),
            )

    class Publisher:
        message = None

        def publish(self, message):
            self.message = message

    header = SimpleNamespace(seq=7, stamp=source_stamp, frame_id="sensor")
    for module, owner in (
        (image_adaptor, image_adaptor.ImageAdaptor),
        (lidar_processor, lidar_processor.LidarBEVProcessor),
    ):
        publisher = Publisher()
        node = SimpleNamespace(bridge=Bridge(), use_ros_time_stamp=True)
        owner.publish_image(
            node,
            publisher,
            np.zeros((1, 1), dtype=np.uint8),
            "mono8",
            header,
            "base_link",
        )
        assert publisher.message.header.stamp is source_stamp


def test_replay_receipt_stamps_do_not_change_camera_source_sync(
    image_adaptor,
    monkeypatch,
):
    class Stamp:
        def __init__(self, seconds):
            self.seconds = seconds

        def to_sec(self):
            return self.seconds

        def to_nsec(self):
            return int(self.seconds * 1_000_000_000)

    node = image_adaptor.ImageAdaptor.__new__(image_adaptor.ImageAdaptor)
    node.use_ros_time_stamp = True
    node.decode_compressed_image = lambda _message: np.zeros(
        (2, 2, 3), dtype=np.uint8
    )
    node.publish_image = lambda *_args, **_kwargs: None
    node.legacy_publishers = {name: object() for name in image_adaptor.CAMERA_NAMES}
    node.model_profiles = {"diffusiondrive_v2"}
    node.lock = threading.Lock()
    node.generation = {name: 0 for name in image_adaptor.CAMERA_NAMES}
    node.latest = {}
    node.last_profile_generation = {
        name: 0 for name in image_adaptor.CAMERA_NAMES
    }
    node.sync_slop = 0.1
    snapshots = []
    node.publish_model_profiles = snapshots.append
    receipt_stamps = iter((Stamp(10.0), Stamp(11.0), Stamp(12.0)))
    monkeypatch.setattr(
        image_adaptor,
        "select_output_stamp",
        lambda _source, _enabled: next(receipt_stamps),
    )
    monkeypatch.setattr(image_adaptor.rospy, "loginfo_throttle", lambda *_args: None)
    monkeypatch.setattr(image_adaptor.rospy, "logwarn_throttle", lambda *_args: None)

    for name, source_seconds in zip(
        image_adaptor.CAMERA_NAMES,
        (1.00, 1.02, 1.01),
    ):
        image_adaptor.ImageAdaptor.image_callback(
            node,
            SimpleNamespace(
                header=SimpleNamespace(
                    seq=1,
                    stamp=Stamp(source_seconds),
                    frame_id=name,
                )
            ),
            name,
        )

    assert len(snapshots) == 1


def test_visualizer_transform_causal_history_and_grid_mask(visualizer):
    rotation = visualizer._quaternion_rotation_matrix([0.0, 0.0, 0.0, 1.0])
    transformed = visualizer.transform_point_xyz(
        [1.0, 2.0, 3.0],
        [2.0, 0.0, 1.5],
        rotation,
    )
    np.testing.assert_allclose(transformed, [3.0, 2.0, 4.5])

    class Stamp:
        def __init__(self, nanoseconds):
            self.nanoseconds = nanoseconds

        def to_nsec(self):
            return self.nanoseconds

    class Header:
        def __init__(self, nanoseconds):
            self.stamp = Stamp(nanoseconds)

    class FakePath:
        def __init__(self, nanoseconds):
            self.header = Header(nanoseconds)

    instance = object.__new__(visualizer.BevVisualizer)
    instance.path_history = deque(maxlen=4)
    instance.path_lock = threading.Lock()
    instance.waypoint_max_age = 2.0

    first = FakePath(1_000_000_000)
    future = FakePath(3_000_000_000)
    instance.waypoints_callback(first)
    instance.waypoints_callback(future)
    assert instance.path_at(Stamp(2_000_000_000)) is first
    assert instance.path_at(Stamp(3_000_000_000)) is future
    assert instance.path_at(Stamp(6_000_000_001)) is None

    instance.height = 2
    instance.width = 3
    instance.lidar_threshold = 1
    instance.lidar_grid_value = 60
    instance.lidar_color = np.asarray([0, 255, 0], dtype=np.uint8)
    _, mask = instance.create_lidar_color_image(
        np.asarray([[0, 1, 60], [255, 60, 2]], dtype=np.uint8)
    )
    assert mask.tolist() == [[False, True, False], [True, False, True]]
