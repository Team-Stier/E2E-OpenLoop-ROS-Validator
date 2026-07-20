"""
pinned InterFuser source model의 neural forward 경계.
입력: 네 camera tensor, LiDAR histogram, CARLA status와 target point.
출력(pub): 없음. Bx10 waypoint를 ModelPrediction으로 반환.
구독(sub): 없음.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from validator_inference.assets import source_checkout_path, verify_runtime_assets
from validator_inference.models.contracts import (
    ModelPrediction,
    ModelRuntimeError,
    model_target_xy,
    prepend_source_path,
    require_module_from_checkout,
    require_inputs,
    require_status,
    ros_image_to_chw,
)


SOURCE_SYMBOL = "timm.models.interfuser.Interfuser.forward"


class _InterFuserNetworkRuntime:
    def __init__(self, profile: dict[str, Any], project_root: Path) -> None:
        checkout = source_checkout_path(profile, project_root)
        vendored_package = checkout / "interfuser"
        prepend_source_path(vendored_package)
        try:
            torch = importlib.import_module("torch")
            models = importlib.import_module("timm.models")
            source_module = importlib.import_module("timm.models.interfuser")
            require_module_from_checkout(source_module, checkout, SOURCE_SYMBOL)
            network = models.create_model("interfuser_baseline")
            checkpoint_path = project_root / profile["runtime_assets"][0]["path"]
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
            network.load_state_dict(checkpoint["state_dict"])
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            network.to(device).eval()
        except Exception as error:
            raise ModelRuntimeError(
                f"model interfuser cannot load {SOURCE_SYMBOL} from {checkout}: "
                f"missing/incompatible dependency, checkpoint, or symbol: {error}"
            ) from error
        self._torch = torch
        self._device = device
        self._network = network

    def tensor(self, value: Any) -> Any:
        if self._torch.is_tensor(value):
            return value.to(self._device, dtype=self._torch.float32)
        return self._torch.as_tensor(
            value, dtype=self._torch.float32, device=self._device
        )

    def image_tensor(self, value: Any) -> Any:
        tensor = self.tensor(ros_image_to_chw(value))
        return tensor.unsqueeze(0) if tensor.ndim == 3 else tensor

    def batch_vector(self, value: Any) -> Any:
        tensor = self.tensor(value)
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

    def forward(self, data: dict[str, Any]) -> Any:
        with self._torch.inference_mode():
            return self._network(data)


class InterFuserModel:
    def __init__(
        self,
        selected_profile: Any,
        project_root: str | Path,
        source_runtime: Any = None,
    ) -> None:
        self.selected_profile = selected_profile
        self.project_root = Path(project_root).resolve()
        self._source_runtime = source_runtime

    def load(self) -> "InterFuserModel":
        if self._source_runtime is not None:
            return self
        try:
            verify_runtime_assets(self.selected_profile.data, self.project_root)
            self._source_runtime = _InterFuserNetworkRuntime(
                self.selected_profile.data, self.project_root
            )
        except Exception as error:
            if isinstance(error, ModelRuntimeError):
                raise
            raise ModelRuntimeError(f"model interfuser load failed: {error}") from error
        return self

    def forward(self, frame: Any, rng_seed: int | None = None) -> ModelPrediction:
        front, left, right, center, lidar, driving, target = require_inputs(
            "interfuser",
            frame,
            (
                "camera_front_224",
                "camera_left_128",
                "camera_right_128",
                "camera_front_center_128",
                "lidar_bev",
                "driving_feature",
                "route_target_point",
            ),
        )
        status = require_status("interfuser", driving, 7)
        self.load()
        runtime = self._source_runtime
        convert = getattr(runtime, "tensor", None)
        tensor = lambda value: convert(value) if convert else value
        image_convert = getattr(runtime, "image_tensor", None)
        vector_convert = getattr(runtime, "batch_vector", None)
        image = lambda value: image_convert(value) if image_convert else value
        vector = lambda value: vector_convert(value) if vector_convert else value
        data = {
            "rgb": image(front),
            "rgb_left": image(left),
            "rgb_right": image(right),
            "rgb_center": image(center),
            "lidar": image(lidar),
            "measurements": vector(status),
            "target_point": vector(
                model_target_xy(
                    target, self.selected_profile.data, carla_y_right=True
                )
            ),
        }
        output = runtime.forward(data)
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise ModelRuntimeError(
                f"model interfuser {SOURCE_SYMBOL} returned no waypoint output"
            )
        return ModelPrediction(trajectory=output[1], source_symbol=SOURCE_SYMBOL)
