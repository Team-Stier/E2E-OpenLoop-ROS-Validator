"""SparseDrive V2 calibration-conditioned source boundary."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from validator_inference.assets import source_checkout_path, verify_runtime_assets
from validator_inference.models.contracts import (
    ModelInputError,
    ModelPrediction,
    ModelRuntimeError,
    prepend_source_path,
    require_module_from_checkout,
    require_inputs,
    require_status,
    ros_image_to_chw,
)


SOURCE_SYMBOL = "navsim.agents.sparsedrive.sparsedrive_model.SparseDriveModel.forward"
SOURCE_CAMERA_ORDER = ("left", "front", "right")
FINAL_IMAGE_WH = (512.0, 256.0)


def _resize_crop_matrix(width: int, height: int) -> np.ndarray:
    final_width, final_height = FINAL_IMAGE_WH
    resize = max(final_height / float(height), final_width / float(width))
    resized_width = int(width * resize)
    resized_height = int(height * resize)
    crop_x = int(max(0, resized_width - final_width) / 2)
    # SparseDriveFeatureBuilder uses bot_pct_lim=(0, 0): keep the bottom of
    # the resized image and remove the full vertical excess from the top.
    crop_y = int(max(0, resized_height - final_height))
    return np.asarray(
        [[resize, 0.0, -crop_x], [0.0, resize, -crop_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def build_projection_conditions(driving_feature: Any) -> dict[str, Any]:
    calibrations = {
        getattr(calibration, "name", ""): calibration
        for calibration in getattr(driving_feature, "camera_calibrations", [])
    }
    if set(calibrations) != {"front", "left", "right"}:
        raise ModelInputError(
            "model sparsedrive_v2 camera_calibrations requires exactly front, left, and right"
        )

    projections = []
    intrinsics = []
    lidar_to_camera = []
    camera_to_lidar = []
    distortion_models = []
    distortion_coefficients = []
    for name in SOURCE_CAMERA_ORDER:
        calibration = calibrations[name]
        try:
            intrinsic = np.asarray(calibration.k_original, dtype=np.float64).reshape(3, 3)
            t_lidar_from_camera = np.asarray(
                calibration.t_lidar_from_camera, dtype=np.float64
            ).reshape(4, 4)
            width = int(calibration.image_width)
            height = int(calibration.image_height)
            distortion_model = str(calibration.distortion_model)
            coefficients = list(calibration.distortion_coefficients)
        except (AttributeError, TypeError, ValueError) as error:
            raise ModelInputError(
                f"model sparsedrive_v2 calibration {name} has malformed projection fields"
            ) from error
        if width <= 0 or height <= 0 or not distortion_model or not coefficients:
            raise ModelInputError(
                f"model sparsedrive_v2 calibration {name} requires image size and distortion"
            )
        t_camera_from_lidar = np.linalg.inv(t_lidar_from_camera)
        resize_crop = _resize_crop_matrix(width, height)
        lidar2img = resize_crop @ intrinsic @ t_camera_from_lidar[:3, :]
        projections.append(lidar2img)
        intrinsics.append(resize_crop @ intrinsic)
        lidar_to_camera.append(t_camera_from_lidar)
        camera_to_lidar.append(t_lidar_from_camera)
        distortion_models.append(distortion_model)
        distortion_coefficients.append(coefficients)

    return {
        "camera_names": list(SOURCE_CAMERA_ORDER),
        "projection_mat": np.stack(projections),
        "cam_intrinsic": np.stack(intrinsics),
        "lidar2cam": np.stack(lidar_to_camera),
        "cam2lidar": np.stack(camera_to_lidar),
        "distortion_models": distortion_models,
        "distortions": distortion_coefficients,
        "image_wh": np.asarray([FINAL_IMAGE_WH] * 3, dtype=np.float32),
    }


class _SparseDriveNetworkRuntime:
    def __init__(self, profile: dict[str, Any], project_root: Path) -> None:
        checkout = source_checkout_path(profile, project_root)
        prepend_source_path(checkout)
        try:
            torch = importlib.import_module("torch")
            config_module = importlib.import_module(
                "navsim.agents.sparsedrive.sparsedrive_config"
            )
            model_module = importlib.import_module(
                "navsim.agents.sparsedrive.sparsedrive_model"
            )
            require_module_from_checkout(model_module, checkout, SOURCE_SYMBOL)
            config = config_module.SparseDriveConfig()
            roles = {
                asset["role"]: project_root / asset["path"]
                for asset in profile["runtime_assets"]
            }
            config.bkb_path = str(roles["image_backbone"])
            config.path_anchor = str(roles["path_vocabulary"])
            config.velocity_anchor = str(roles["velocity_vocabulary"])
            config.trajectory_anchor = str(roles["trajectory_vocabulary"])
            network = model_module.SparseDriveModel(config)
            checkpoint = torch.load(str(roles["checkpoint"]), map_location="cpu")
            raw_state = checkpoint.get("state_dict", checkpoint)
            state = {}
            for key, value in raw_state.items():
                for prefix in (
                    "agent._sparsedrive_model.",
                    "_sparsedrive_model.",
                    "agent.",
                ):
                    if key.startswith(prefix):
                        key = key[len(prefix) :]
                        break
                state[key] = value
            network.load_state_dict(state, strict=False)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            network.to(device).eval()
        except Exception as error:
            raise ModelRuntimeError(
                f"model sparsedrive_v2 cannot load {SOURCE_SYMBOL} from {checkout}: "
                f"missing/incompatible dependency, checkpoint, CUDA op, or symbol: {error}"
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

    def prepare_images(self, values: list[Any]) -> Any:
        tensors = [self.tensor(ros_image_to_chw(value)) for value in values]
        tensors = [value.squeeze(0) if value.ndim == 4 and value.shape[0] == 1 else value for value in tensors]
        return self._torch.stack(tensors, dim=0).unsqueeze(0)

    def forward(self, features: dict[str, Any]) -> Any:
        with self._torch.inference_mode():
            return self._network(features, targets={})


class SparseDriveV2Model:
    def __init__(
        self,
        selected_profile: Any,
        project_root: str | Path,
        source_runtime: Any = None,
    ) -> None:
        self.selected_profile = selected_profile
        self.project_root = Path(project_root).resolve()
        self._source_runtime = source_runtime

    def load(self) -> "SparseDriveV2Model":
        if self._source_runtime is not None:
            return self
        try:
            verify_runtime_assets(self.selected_profile.data, self.project_root)
            self._source_runtime = _SparseDriveNetworkRuntime(
                self.selected_profile.data, self.project_root
            )
        except Exception as error:
            if isinstance(error, ModelRuntimeError):
                raise
            raise ModelRuntimeError(
                f"model sparsedrive_v2 load failed: {error}"
            ) from error
        return self

    def forward(self, frame: Any, rng_seed: int | None = None) -> ModelPrediction:
        left, front, right, driving = require_inputs(
            "sparsedrive_v2",
            frame,
            ("camera_left", "camera_main", "camera_right", "driving_feature"),
        )
        status = require_status("sparsedrive_v2", driving, 8)
        conditions = build_projection_conditions(driving)
        self.load()
        runtime = self._source_runtime
        prepare_images = getattr(runtime, "prepare_images", None)
        conditions["imgs"] = (
            prepare_images([left, front, right])
            if prepare_images
            else [left, front, right]
        )
        convert = getattr(runtime, "tensor", None)
        if convert:
            for key in (
                "projection_mat",
                "cam_intrinsic",
                "lidar2cam",
                "cam2lidar",
                "image_wh",
            ):
                conditions[key] = convert(conditions[key]).unsqueeze(0)
            status = convert(status).unsqueeze(0)
        output = runtime.forward(
            {"camera_feature": conditions, "status_feature": status}
        )
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, dict) or output.get("trajectory") is None:
            raise ModelRuntimeError(
                f"model sparsedrive_v2 {SOURCE_SYMBOL} returned no trajectory"
            )
        return ModelPrediction(
            trajectory=output["trajectory"], source_symbol=SOURCE_SYMBOL
        )
