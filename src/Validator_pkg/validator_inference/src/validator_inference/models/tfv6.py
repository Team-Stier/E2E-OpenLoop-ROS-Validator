"""LEAD TFv6 official ``noradar_resnet34`` source boundary."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from validator_inference.assets import source_checkout_path, verify_runtime_assets
from validator_inference.models.contracts import (
    ModelInputError,
    ModelPrediction,
    ModelRuntimeError,
    point_xy,
    prepend_source_path,
    require_module_from_checkout,
    require_inputs,
    require_status,
    ros_image_to_chw,
)


SOURCE_SYMBOL = "lead.inference.open_loop_inference.OpenLoopInference.forward"
TEMPORAL_LIDAR_SYMBOL = "lead.data_loader.carla_dataset_utils.rasterize_lidar"


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
            require_module_from_checkout(inference_module, checkout, SOURCE_SYMBOL)
            require_module_from_checkout(lidar_module, checkout, TEMPORAL_LIDAR_SYMBOL)
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
        self._model = inference_module.OpenLoopInference(
            config_training=training_config,
            config_open_loop=open_config,
            model_path=str(config_path.parent),
            device=torch.device("cuda:0"),
            prefix="model",
        )

    def build_temporal_lidar(self, history: list[Any]) -> Any:
        """Merge ten source-aligned sweeps and invoke LEAD's official rasterizer.

        Each engine history item must be an ``(points_xyz, T_anchor_from_sweep)``
        pair, or expose those names.  This keeps the source transform explicit and
        prevents silently treating ten differently posed sweeps as one cloud.
        """
        np = importlib.import_module("numpy")
        merged = []
        for item in history:
            if isinstance(item, tuple) and len(item) == 2:
                points, transform = item
            else:
                points = getattr(item, "points_xyz", None)
                transform = getattr(item, "t_anchor_from_sweep", None)
            if points is None or transform is None:
                raise ModelRuntimeError(
                    "model tfv6 temporal LiDAR requires each lidar_raw_history item "
                    "to carry points_xyz and T_anchor_from_sweep"
                )
            xyz = np.asarray(points, dtype=np.float32)[:, :3]
            matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
            homogeneous = np.concatenate(
                [xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1
            )
            aligned = (matrix @ homogeneous.T).T[:, :3]
            aligned[:, 1] *= -1.0
            merged.append(aligned)
        points = np.concatenate(merged, axis=0)
        raster = self._rasterize_lidar(self._training_config, points)
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
        previous_target = point_xy(previous_target, carla_y_right=True)
        current_target = point_xy(current_target, carla_y_right=True)
        next_target = point_xy(next_target, carla_y_right=True)

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
