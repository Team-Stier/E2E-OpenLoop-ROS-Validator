"""
PDM 평가를 제거하고 learned selector만 실행하는 DiffusionDrive V2 경계.
입력: stitched camera, LiDAR BEV, NAVSIM driving feature, local anchor/checkpoint.
출력(pub): 없음. 선택된 Bx8 trajectory를 ModelPrediction으로 반환.
구독(sub): 없음.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from validator_inference.assets import (
    DIFFUSION_DERIVATIVE_RELATIVE_PATH,
    selector_flow_stages,
    source_checkout_path,
    verify_runtime_assets,
)
from validator_inference.models.contracts import (
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


SOURCE_SYMBOL = (
    "navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel_inference."
    "V2TransfuserModel.forward"
)

# PDM-free derivative는 이 evaluation vocabulary parameter/buffer만 의도적으로
# 삭제한다. 그 외 모든 official checkpoint key는 strict-match해야 한다.
PDM_REMOVED_CHECKPOINT_KEY = "_trajectory_head.vocab"


def load_diffusiondrive_selector_state_dict(
    network: Any,
    raw_state: dict[str, Any],
    *,
    checkpoint_path: str | Path,
) -> None:
    """official selector checkpoint를 PDM-free derivative에 strict-load한다."""
    normalized: dict[str, Any] = {}
    for raw_key, value in raw_state.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("agent._transfuser_model."):
            key = key[len("agent._transfuser_model.") :]
        if key == PDM_REMOVED_CHECKPOINT_KEY:
            continue
        if key in normalized:
            raise SourceRuntimeError(
                f"model diffusiondrive_v2 checkpoint {checkpoint_path} has duplicate "
                f"normalized key {key}"
            )
        normalized[key] = value

    load_source_state_dict(
        network,
        normalized,
        model_id="diffusiondrive_v2",
        checkpoint_path=checkpoint_path,
        documented_prefixes=(),
    )


class _DiffusionDriveSelectorRuntime:
    def __init__(self, profile: dict[str, Any], project_root: Path) -> None:
        checkout = source_checkout_path(profile, project_root)
        derivative = checkout / DIFFUSION_DERIVATIVE_RELATIVE_PATH
        if not derivative.is_file():
            raise ModelRuntimeError(
                f"model diffusiondrive_v2 missing generated PDM-free source derivative: {derivative}; "
                "run fetch_model_assets.py"
            )
        derivative_text = derivative.read_text(encoding="utf-8")
        forbidden_tokens = (
            "pdm_planner",
            "navsim.evaluate.pdm_score",
            "ProcessPoolExecutor",
            "_pdm_pool",
            "metric_cache",
            "get_pdm_score_para",
            "reward_dict",
        )
        present = [token for token in forbidden_tokens if token in derivative_text]
        if present:
            raise ModelRuntimeError(
                f"model diffusiondrive_v2 derivative {derivative} contains forbidden "
                f"PDM runtime token(s): {', '.join(present)}"
            )
        if 'return {"trajectory": traj_to_score[:, -1]}' not in derivative_text:
            raise ModelRuntimeError(
                f"model diffusiondrive_v2 derivative {derivative} lacks final learned selector return"
            )
        try:
            selector_flow_stages(derivative_text)
        except Exception as error:
            raise ModelRuntimeError(
                f"model diffusiondrive_v2 derivative {derivative} violates learned "
                f"selector structure: {error}"
            ) from error

        prepend_source_path(checkout)
        try:
            torch = importlib.import_module("torch")
            config_module = importlib.import_module(
                "navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config"
            )
            model_module = importlib.import_module(
                "navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel_inference"
            )
            require_module_from_checkout(model_module, checkout, SOURCE_SYMBOL)
            config = config_module.TransfuserConfig()
            roles = {
                asset["role"]: project_root / asset["path"]
                for asset in profile["runtime_assets"]
            }
            config.bkb_path = str(roles["image_backbone"])
            config.plan_anchor_path = str(roles["trajectory_anchor"])
            network = model_module.V2TransfuserModel(config)
            checkpoint = torch.load(str(roles["checkpoint"]), map_location="cpu")
            raw_state = checkpoint.get("state_dict", checkpoint)
            load_diffusiondrive_selector_state_dict(
                network,
                raw_state,
                checkpoint_path=roles["checkpoint"],
            )
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            network.to(device).eval()
        except SourceRuntimeError:
            raise
        except Exception as error:
            raise ModelRuntimeError(
                f"model diffusiondrive_v2 cannot load {SOURCE_SYMBOL} from {checkout}: "
                f"missing/incompatible dependency, checkpoint, CUDA op, or symbol: {error}"
            ) from error
        self._torch = torch
        self._device = device
        self._network = network

    def set_seed(self, seed: int) -> None:
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)

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

    def forward_pdm_free(self, features: dict[str, Any]) -> dict[str, Any]:
        with self._torch.inference_mode():
            return self._network(features)


class DiffusionDriveV2InferenceModel:
    def __init__(
        self,
        selected_profile: Any,
        project_root: str | Path,
        source_runtime: Any = None,
    ) -> None:
        self.selected_profile = selected_profile
        self.project_root = Path(project_root).resolve()
        self._source_runtime = source_runtime

    def load(self) -> "DiffusionDriveV2InferenceModel":
        if self._source_runtime is not None:
            return self
        try:
            verify_runtime_assets(self.selected_profile.data, self.project_root)
            self._source_runtime = _DiffusionDriveSelectorRuntime(
                self.selected_profile.data, self.project_root
            )
        except Exception as error:
            if isinstance(error, ModelRuntimeError):
                raise
            raise ModelRuntimeError(
                f"model diffusiondrive_v2 load failed: {error}"
            ) from error
        return self

    def forward(self, frame: Any, rng_seed: int) -> ModelPrediction:
        camera, lidar, driving = require_inputs(
            "diffusiondrive_v2",
            frame,
            ("camera_main", "lidar_bev", "driving_feature"),
        )
        status = require_status("diffusiondrive_v2", driving, 8)
        self.load()
        runtime = self._source_runtime
        runtime.set_seed(int(rng_seed))
        convert = getattr(runtime, "tensor", None)
        tensor = lambda value: convert(value) if convert else value
        image_convert = getattr(runtime, "image_tensor", None)
        vector_convert = getattr(runtime, "batch_vector", None)
        output = runtime.forward_pdm_free(
            {
                "camera_feature": image_convert(camera) if image_convert else camera,
                "lidar_feature": image_convert(lidar) if image_convert else lidar,
                "status_feature": vector_convert(status) if vector_convert else status,
            }
        )
        if not isinstance(output, dict) or output.get("trajectory") is None:
            raise ModelRuntimeError(
                f"model diffusiondrive_v2 {SOURCE_SYMBOL} returned no trajectory"
            )
        return ModelPrediction(
            trajectory=output["trajectory"], source_symbol=SOURCE_SYMBOL
        )
