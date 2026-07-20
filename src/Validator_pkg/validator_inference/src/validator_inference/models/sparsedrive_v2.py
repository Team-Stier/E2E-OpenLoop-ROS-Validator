"""
inline calibration projection condition을 사용하는 SparseDrive V2 source 경계.
입력: left/front/right image, NAVSIM driving feature, profile calibration.
출력(pub): 없음. Bx8 trajectory를 ModelPrediction으로 반환.
구독(sub): 없음.
"""

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
    SourceRuntimeError,
    load_source_state_dict,
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
    """원본 camera pixel 좌표를 SparseDrive 최종 crop 좌표로 옮기는 행렬을 만든다.

    입력: profile이 선언한 원본 image width와 height.
    반환: 512×288 resize 뒤 y=32 crop을 합성한 3×3 matrix.
    """
    if width <= 0 or height <= 0:
        raise ModelInputError("SparseDrive image dimensions must be positive")
    # pinned builder는 1920x1080 canonical image를 512x288로 resize한 뒤
    # y=[32, 288]을 crop한다. Preprocess가 각 raw camera를 이 canonical
    # geometry로 제공하므로 raw pixel에 aspect-preserving resize를 적용하지 않고
    # 두 단계를 합성한다.
    scale_x = 512.0 / float(width)
    scale_y = 288.0 / float(height)
    return np.asarray(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, -32.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def build_projection_conditions(driving_feature: Any) -> dict[str, Any]:
    """inline calibration으로 SparseDrive image projection condition을 구성한다.

    입력: front·left·right ``camera_calibrations``를 포함한 driving feature.
    반환: source가 요구하는 projection, intrinsic, extrinsic, distortion dict.
    """
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
    """pinned SparseDrive network과 calibration/image tensor adapter를 감싼다.

    입력: SparseDrive profile과 project root.
    반환: wrapper가 source-native forward에 사용할 runtime adapter.
    """

    def __init__(self, profile: dict[str, Any], project_root: Path) -> None:
        """원본 config·backbone·vocabulary·checkpoint를 strict-load한다.

        입력: 검증된 SparseDrive profile과 project root.
        부작용: source import path를 설정하고 eval network를 device에 올린다.
        """
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
            load_source_state_dict(
                network,
                raw_state,
                model_id="sparsedrive_v2",
                checkpoint_path=roles["checkpoint"],
                documented_prefixes=(
                    "agent._sparsedrive_model.",
                    "_sparsedrive_model.",
                ),
            )
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            network.to(device).eval()
        except SourceRuntimeError:
            raise
        except Exception as error:
            raise ModelRuntimeError(
                f"model sparsedrive_v2 cannot load {SOURCE_SYMBOL} from {checkout}: "
                f"missing/incompatible dependency, checkpoint, CUDA op, or symbol: {error}"
            ) from error
        self._torch = torch
        self._device = device
        self._network = network

    def tensor(self, value: Any) -> Any:
        """수치 condition을 runtime device의 float32 tensor로 변환한다.

        입력: tensor 또는 numpy/sequence 값.
        반환: CPU/CUDA float32 tensor.
        """
        if self._torch.is_tensor(value):
            return value.to(self._device, dtype=self._torch.float32)
        return self._torch.as_tensor(
            value, dtype=self._torch.float32, device=self._device
        )

    def prepare_images(self, values: list[Any]) -> Any:
        """left·front·right ROS image를 SparseDrive batch image tensor로 묶는다.

        입력: source camera 순서의 image 세 개.
        반환: shape ``[1, 3, C, H, W]`` image tensor.
        """
        tensors = [self.tensor(ros_image_to_chw(value)) for value in values]
        tensors = [value.squeeze(0) if value.ndim == 4 and value.shape[0] == 1 else value for value in tensors]
        return self._torch.stack(tensors, dim=0).unsqueeze(0)

    def forward(self, features: dict[str, Any]) -> Any:
        """gradient 없이 source SparseDrive network를 open-loop 모드로 실행한다.

        입력: ``camera_feature``와 ``status_feature`` source feature dict.
        반환: ``trajectory``를 포함해야 하는 원본 output.
        """
        with self._torch.inference_mode():
            return self._network(features, targets={})


class SparseDriveV2Model:
    """inline calibration SparseDrive V2 입력을 공통 prediction 계약으로 변환한다.

    입력: selected profile, project root, 선택적 source runtime.
    반환: source trajectory를 담은 ``ModelPrediction``.
    """

    def __init__(
        self,
        selected_profile: Any,
        project_root: str | Path,
        source_runtime: Any = None,
    ) -> None:
        """lazy-load할 SparseDrive V2 wrapper 상태를 초기화한다.

        입력: 선택 profile, workspace root, 선택적으로 주입된 runtime.
        부작용: 실제 model load를 첫 forward까지 지연한다.
        """
        self.selected_profile = selected_profile
        self.project_root = Path(project_root).resolve()
        self._source_runtime = source_runtime

    def load(self) -> "SparseDriveV2Model":
        """asset을 검증하고 아직 없으면 strict source runtime을 load한다.

        입력: 없음.
        반환: load된 자기 자신.
        부작용: checkpoint와 source dependency를 검증하고 network를 초기화한다.
        """
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
        """세 camera, status, inline calibration을 SparseDrive forward에 전달한다.

        입력: left/front/right image와 8-value driving feature frame, 미사용 seed.
        반환: source ``trajectory``를 담은 ``ModelPrediction``.
        """
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
