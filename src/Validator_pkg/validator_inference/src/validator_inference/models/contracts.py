"""Small result/error contracts shared by concrete model runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, Tuple

import numpy as np


class ModelRuntimeError(RuntimeError):
    """A selected source/checkpoint runtime cannot be loaded or invoked."""


class ModelInputError(ValueError):
    """An inference frame violates the selected source model's input contract."""


@dataclass(frozen=True)
class ModelPrediction:
    trajectory: Any
    source_symbol: str


def require_inputs(model_id: str, frame: Any, names: Iterable[str]) -> Tuple[Any, ...]:
    missing = [name for name in names if name not in frame.inputs]
    if missing:
        raise ModelInputError(
            f"model {model_id} missing input(s): {', '.join(missing)}"
        )
    return tuple(frame.inputs[name] for name in names)


def require_status(model_id: str, driving_feature: Any, length: int) -> list[float]:
    try:
        status = list(driving_feature.status_feature)
    except AttributeError as error:
        raise ModelInputError(
            f"model {model_id} driving_feature missing status_feature"
        ) from error
    if len(status) != length:
        raise ModelInputError(
            f"model {model_id} status_feature requires length {length}, got {len(status)}"
        )
    return status


def prepend_source_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def require_module_from_checkout(module: Any, checkout: Path, symbol: str) -> None:
    module_path = Path(getattr(module, "__file__", "")).resolve()
    checkout = checkout.resolve()
    if not module_path.is_file() or checkout not in module_path.parents:
        raise ModelRuntimeError(
            f"source symbol {symbol} resolved outside pinned checkout {checkout}: "
            f"{module_path}"
        )


def ros_image_to_chw(value: Any) -> Any:
    """Decode a ROS ``sensor_msgs/Image`` without importing cv_bridge."""
    if not all(hasattr(value, name) for name in ("height", "width", "encoding", "data")):
        return value
    encoding = str(value.encoding).lower()
    encodings = {
        "mono8": (np.uint8, 1),
        "8uc1": (np.uint8, 1),
        "rgb8": (np.uint8, 3),
        "bgr8": (np.uint8, 3),
        "8uc3": (np.uint8, 3),
        "rgba8": (np.uint8, 4),
        "bgra8": (np.uint8, 4),
        "8uc4": (np.uint8, 4),
        "32fc1": (np.float32, 1),
        "32fc2": (np.float32, 2),
        "32fc3": (np.float32, 3),
        "32fc4": (np.float32, 4),
    }
    if encoding not in encodings:
        raise ModelInputError(f"unsupported ROS image encoding: {value.encoding}")
    dtype, channels = encodings[encoding]
    dtype = np.dtype(dtype).newbyteorder(">" if getattr(value, "is_bigendian", 0) else "<")
    raw = np.frombuffer(bytes(value.data), dtype=dtype)
    row_items = int(value.step) // dtype.itemsize
    expected_row_items = int(value.width) * channels
    if row_items < expected_row_items or raw.size < int(value.height) * row_items:
        raise ModelInputError(
            f"ROS image {value.encoding} has invalid step/data length for "
            f"{value.width}x{value.height}"
        )
    rows = raw[: int(value.height) * row_items].reshape(int(value.height), row_items)
    image = rows[:, :expected_row_items]
    if channels == 1:
        image = image.reshape(int(value.height), int(value.width))[None, ...]
    else:
        image = image.reshape(int(value.height), int(value.width), channels)
        if encoding.startswith("bgr"):
            image = image[..., [2, 1, 0] + list(range(3, channels))]
        image = image.transpose(2, 0, 1)
    return np.ascontiguousarray(image.astype(dtype.newbyteorder("="), copy=False))


def point_xy(value: Any, *, carla_y_right: bool) -> Any:
    if hasattr(value, "point"):
        x = float(value.point.x)
        y = float(value.point.y)
        return [x, -y if carla_y_right else y]
    return value
