"""Dynamic producer/visualizer contract checks with synthetic sensor data."""

from collections import deque
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import threading

import pytest


np = pytest.importorskip("numpy")

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

    diffusion = lidar_processor.prepare_diffusiondrive_v2(points)
    assert diffusion.shape == (256, 256)
    assert diffusion.dtype == np.float32
    assert 0.0 <= float(diffusion.min()) <= float(diffusion.max()) <= 1.0


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
