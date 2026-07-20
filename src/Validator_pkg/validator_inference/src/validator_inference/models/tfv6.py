"""LEAD TFv6 official ``noradar_resnet34`` source boundary."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from validator_inference.assets import source_checkout_path, verify_runtime_assets
from validator_inference.models.contracts import (
    ModelInputError,
    ModelPrediction,
    ModelRuntimeError,
    model_target_xy,
    prepend_source_path,
    require_module_from_checkout,
    require_inputs,
    require_status,
    ros_image_to_chw,
)


SOURCE_SYMBOL = "lead.inference.open_loop_inference.OpenLoopInference.forward"
TEMPORAL_LIDAR_SYMBOL = "lead.data_loader.carla_dataset_utils.rasterize_lidar"
GROUND_REMOVAL_SYMBOL = "lead.common.ransac.remove_ground"


def _rigid_transform_matrix(declaration: dict[str, Any]) -> np.ndarray:
    try:
        translation = np.asarray(declaration["translation_m"], dtype=np.float64)
        quaternion = np.asarray(declaration["quaternion_xyzw"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ModelRuntimeError(
            "model tfv6 requires full T_base_from_lidar translation/quaternion"
        ) from error
    if translation.shape != (3,) or quaternion.shape != (4,):
        raise ModelRuntimeError(
            "model tfv6 T_base_from_lidar requires translation[3] and quaternion[4]"
        )
    x, y, z, w = quaternion
    norm = float(np.dot(quaternion, quaternion))
    if norm <= np.finfo(float).eps:
        raise ModelRuntimeError("model tfv6 T_base_from_lidar quaternion is zero")
    scale = 2.0 / norm
    matrix = np.asarray(
        [
            [
                1.0 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
                translation[0],
            ],
            [
                scale * (x * y + z * w),
                1.0 - scale * (x * x + z * z),
                scale * (y * z - x * w),
                translation[1],
            ],
            [
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1.0 - scale * (x * x + y * y),
                translation[2],
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return matrix


def _transform_xyz(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1
    )
    return (transform @ homogeneous.T).T[:, :3].astype(np.float32)


def _source_prefilter_tfv6_sweep(
    points_base_ros: np.ndarray,
    *,
    expert_config: Any,
    remove_ground: Callable[..., Any],
) -> np.ndarray:
    """Port LEAD base_agent.py:158-163 and 203-237 without heuristics."""
    lidar_x = points_base_ros[:, 0]
    lidar_y = points_base_ros[:, 1]
    points = points_base_ros[
        (np.abs(lidar_x) > expert_config.ego_extent_x)
        & (np.abs(lidar_y) > expert_config.ego_extent_y)
    ]

    if expert_config.save_only_non_ground_lidar and len(points):
        ground_mask = np.asarray(
            remove_ground(points, expert_config, parallel=True), dtype=bool
        )
        if ground_mask.shape != (points.shape[0],):
            raise ModelRuntimeError(
                "model tfv6 pinned RANSAC returned an invalid ground mask"
            )
        points = points[~ground_mask]

    if expert_config.save_lidar_only_inside_bev:
        lidar_x = points[:, 0]
        lidar_y = points[:, 1]
        lidar_z = points[:, 2]
        inside = (
            (expert_config.min_x_meter <= lidar_x)
            & (lidar_x <= expert_config.max_x_meter)
            & (expert_config.min_y_meter <= lidar_y)
            & (lidar_y <= expert_config.max_y_meter)
            & (expert_config.min_height_lidar <= lidar_z)
            & (lidar_z <= expert_config.max_height_lidar)
        )
        points = points[inside]
    return points


def prepare_tfv6_temporal_lidar(
    history: list[Any],
    *,
    training_config: Any,
    expert_config: Any,
    T_base_from_lidar: dict[str, Any],
    point_precision_xyz: tuple[float, float, float],
    read_points: Callable[..., Any],
    remove_ground: Callable[..., Any],
    rasterize_lidar: Callable[..., Any],
    compress_float_image: Callable[..., Any],
    decompress_float_image: Callable[..., Any],
) -> np.ndarray:
    """Reproduce pinned LEAD prefilter, accumulation, raster and PNG stages.

    ROS sensor points remain x-forward/y-left through mount transform,
    per-sweep filtering, and newest-anchor alignment.  The conversion to the
    pinned source's CARLA y-right convention happens exactly once immediately
    before the source quantization/raster boundary.
    """
    base_from_lidar = _rigid_transform_matrix(T_base_from_lidar)
    merged = []
    for item in history:
        point_cloud = getattr(item, "point_cloud", None)
        transform = getattr(
            item, "T_anchor_lidar_from_sweep_lidar", None
        )
        if point_cloud is None or transform is None:
            raise ModelRuntimeError(
                "model tfv6 temporal LiDAR requires point_cloud and "
                "T_anchor_lidar_from_sweep_lidar for every sweep"
            )
        points = np.asarray(
            list(
                read_points(
                    point_cloud,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )
            ),
            dtype=np.float32,
        )
        if points.size == 0:
            points = np.zeros((0, 3), dtype=np.float32)
        else:
            points = points.reshape(-1, 3)
            points = points[np.isfinite(points).all(axis=1)]
        # Source-equivalent order from lead/common/base_agent.py:140-237:
        # sensor->ego, ego footprint, exact RANSAC, then BEV/height bounds.
        points_base = _transform_xyz(points, base_from_lidar)
        points_base = _source_prefilter_tfv6_sweep(
            points_base,
            expert_config=expert_config,
            remove_ground=remove_ground,
        )
        # InputBuffer pairs map-to-base poses, so this maps sweep base/ego
        # coordinates into the newest anchor base/ego coordinates.
        matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        aligned = _transform_xyz(points_base, matrix)
        merged.append(aligned)

    points = np.concatenate(merged, axis=0)
    # ROS x-forward/y-left -> pinned CARLA x-forward/y-right, exactly once.
    points[:, 1] *= -1.0
    # lead/inference/sensor_agent.py:386-413
    for axis, precision in enumerate(point_precision_xyz):
        points[:, axis] = np.round(points[:, axis] / precision) * precision
    raster = rasterize_lidar(config=training_config, lidar=points)[..., None]
    compressed = compress_float_image(raster, training_config)
    return np.asarray(decompress_float_image(compressed), dtype=np.float32).squeeze()


class _LeadOpenLoopRuntime:
    def __init__(self, profile: dict[str, Any], project_root: Path) -> None:
        checkout = source_checkout_path(profile, project_root)
        prepend_source_path(checkout)
        try:
            torch = importlib.import_module("torch")
            config_module = importlib.import_module("lead.training.config_training")
            open_config_module = importlib.import_module("lead.inference.config_open_loop")
            inference_module = importlib.import_module("lead.inference.open_loop_inference")
            lidar_module = importlib.import_module("lead.data_loader.carla_dataset_utils")
            cache_module = importlib.import_module("lead.data_loader.training_cache")
            ransac_module = importlib.import_module("lead.common.ransac")
            expert_config_module = importlib.import_module("lead.expert.config_expert")
            point_cloud_module = importlib.import_module("sensor_msgs.point_cloud2")
            require_module_from_checkout(inference_module, checkout, SOURCE_SYMBOL)
            require_module_from_checkout(lidar_module, checkout, TEMPORAL_LIDAR_SYMBOL)
            require_module_from_checkout(ransac_module, checkout, GROUND_REMOVAL_SYMBOL)
        except Exception as error:
            raise ModelRuntimeError(
                f"model tfv6 cannot import {SOURCE_SYMBOL} from {checkout}: "
                f"missing/incompatible dependency or symbol: {error}"
            ) from error
        if not torch.cuda.is_available():
            raise ModelRuntimeError(
                "model tfv6 source OpenLoopInference requires CUDA for checkpoint loading"
            )

        roles = {asset["role"]: project_root / asset["path"] for asset in profile["runtime_assets"]}
        config_path = roles["checkpoint_config"]
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        training_config = config_module.TrainingConfig(
            loaded, raise_error_on_missing_key=False
        )
        open_config = open_config_module.OpenLoopConfig(
            raise_error_on_missing_key=False
        )
        self._torch = torch
        self._training_config = training_config
        self._rasterize_lidar = lidar_module.rasterize_lidar
        self._compress_float_image = cache_module.compress_float_image
        self._decompress_float_image = cache_module.decompress_float_image
        self._read_points = point_cloud_module.read_points
        self._remove_ground = ransac_module.remove_ground
        expert_config = expert_config_module.ExpertConfig()
        self._expert_config = expert_config
        self._T_base_from_lidar = profile["static_transforms"]["T_base_from_lidar"]
        self._point_precision_xyz = (
            float(expert_config.point_precision_x),
            float(expert_config.point_precision_y),
            float(expert_config.point_precision_z),
        )
        self._model = inference_module.OpenLoopInference(
            config_training=training_config,
            config_open_loop=open_config,
            model_path=str(config_path.parent),
            device=torch.device("cuda:0"),
            prefix="model",
        )

    def build_temporal_lidar(self, history: list[Any]) -> Any:
        raster = prepare_tfv6_temporal_lidar(
            history,
            training_config=self._training_config,
            expert_config=self._expert_config,
            T_base_from_lidar=self._T_base_from_lidar,
            point_precision_xyz=self._point_precision_xyz,
            read_points=self._read_points,
            remove_ground=self._remove_ground,
            rasterize_lidar=self._rasterize_lidar,
            compress_float_image=self._compress_float_image,
            decompress_float_image=self._decompress_float_image,
        )
        return self._torch.as_tensor(raster, dtype=self._torch.float32, device="cuda")[
            None, None
        ]

    def tensor(self, value: Any) -> Any:
        if self._torch.is_tensor(value):
            return value.to(device="cuda", dtype=self._torch.float32)
        return self._torch.as_tensor(value, dtype=self._torch.float32, device="cuda")

    def image_tensor(self, value: Any) -> Any:
        tensor = self.tensor(ros_image_to_chw(value))
        return tensor.unsqueeze(0) if tensor.ndim == 3 else tensor

    def batch_vector(self, value: Any) -> Any:
        tensor = self.tensor(value)
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

    def forward(self, data: dict[str, Any]) -> Any:
        return self._model.forward(data)


class TFv6Model:
    """Concrete wrapper for the pinned LEAD open-loop ensemble."""

    def __init__(
        self,
        selected_profile: Any,
        project_root: str | Path,
        source_runtime: Any = None,
    ) -> None:
        self.selected_profile = selected_profile
        self.project_root = Path(project_root).resolve()
        self._source_runtime = source_runtime

    def load(self) -> "TFv6Model":
        if self._source_runtime is not None:
            return self
        try:
            verify_runtime_assets(self.selected_profile.data, self.project_root)
            self._source_runtime = _LeadOpenLoopRuntime(
                self.selected_profile.data, self.project_root
            )
        except Exception as error:
            if isinstance(error, ModelRuntimeError):
                raise
            raise ModelRuntimeError(f"model tfv6 load failed: {error}") from error
        return self

    def forward(self, frame: Any, rng_seed: int | None = None) -> ModelPrediction:
        (
            camera,
            history,
            driving,
            previous_target,
            current_target,
            next_target,
        ) = require_inputs(
            "tfv6",
            frame,
            (
                "camera_main",
                "lidar_raw_history",
                "driving_feature",
                "route_target_point_previous",
                "route_target_point",
                "route_target_point_next",
            ),
        )
        if len(history) != 10:
            raise ModelInputError(
                f"model tfv6 lidar_raw_history requires exactly 10 sweeps, got {len(history)}"
            )
        status = require_status("tfv6", driving, 7)
        radar = list(getattr(driving, "tfv6_radar", []))
        rows = int(getattr(driving, "tfv6_radar_rows", 0))
        cols = int(getattr(driving, "tfv6_radar_cols", 0))
        if radar or rows != 0 or cols != 0:
            raise ModelInputError(
                "model tfv6 official noradar_resnet34 requires radar fields to be empty "
                "with rows=0 and cols=0"
            )
        self.load()
        runtime = self._source_runtime
        rasterized = runtime.build_temporal_lidar(history)

        def tensor(value: Any) -> Any:
            convert = getattr(runtime, "tensor", None)
            return convert(value) if convert else value

        image_convert = getattr(runtime, "image_tensor", None)
        vector_convert = getattr(runtime, "batch_vector", None)
        previous_target = model_target_xy(
            previous_target, self.selected_profile.data, carla_y_right=True
        )
        current_target = model_target_xy(
            current_target, self.selected_profile.data, carla_y_right=True
        )
        next_target = model_target_xy(
            next_target, self.selected_profile.data, carla_y_right=True
        )

        data = {
            "rgb": image_convert(camera) if image_convert else camera,
            "rasterized_lidar": rasterized,
            "target_point_previous": vector_convert(previous_target) if vector_convert else previous_target,
            "target_point": vector_convert(current_target) if vector_convert else current_target,
            "target_point_next": vector_convert(next_target) if vector_convert else next_target,
            "speed": tensor([status[6]]),
            "command": vector_convert(status[:6]) if vector_convert else status[:6],
            "next_command": vector_convert(status[:6]) if vector_convert else status[:6],
        }
        output = runtime.forward(data)
        trajectory = getattr(output, "pred_future_waypoints", None)
        if trajectory is None:
            raise ModelRuntimeError(
                f"model tfv6 {SOURCE_SYMBOL} returned no pred_future_waypoints"
            )
        return ModelPrediction(trajectory=trajectory, source_symbol=SOURCE_SYMBOL)
