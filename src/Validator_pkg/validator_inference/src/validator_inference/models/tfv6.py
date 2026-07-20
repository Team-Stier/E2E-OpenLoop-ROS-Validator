"""
공식 LEAD TFv6 noradar_resnet34 ensemble의 source forward 경계.
입력: panorama, pose-aligned 10-sweep LiDAR, status, 세 target point.
출력(pub): 없음. 8개 CARLA-relative waypoint를 ModelPrediction으로 반환.
구독(sub): 없음.
"""

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
    """profile의 ``T_base_from_lidar``를 homogeneous transform으로 검증·변환한다.

    입력: translation과 ``xyzw`` quaternion을 가진 static transform 선언.
    반환: LiDAR 점을 base 좌표로 옮기는 4×4 float64 matrix.
    """
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
    """XYZ point batch에 하나의 homogeneous rigid transform을 적용한다.

    입력: ``N×3`` points와 ``4×4`` transform.
    반환: 같은 순서를 유지한 ``N×3`` float32 transformed points.
    """
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
    """heuristic 없이 LEAD의 ego mask·RANSAC·BEV bounds 순서를 port한다.

    입력: ROS base 좌표 points, 원본 expert config, 원본 ground remover.
    반환: source rasterizer에 전달할 정제된 base 좌표 point batch.
    """
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
    """pinned LEAD의 prefilter, accumulation, raster, PNG 단계를 재현한다.

    입력: pose-align된 10-sweep history, 원본 config·point reader·raster helpers.
    반환: source ``OpenLoopInference``가 기대하는 float32 LiDAR raster.
    좌표: ROS x-forward/y-left를 mount·filter·anchor alignment 뒤 raster 직전에만
    pinned CARLA x-forward/y-right로 한 번 변환한다.
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
        # lead/common/base_agent.py:140-237과 같은 순서:
        # sensor->ego, ego footprint, 정확한 RANSAC, 그 다음 BEV/height bounds.
        points_base = _transform_xyz(points, base_from_lidar)
        points_base = _source_prefilter_tfv6_sweep(
            points_base,
            expert_config=expert_config,
            remove_ground=remove_ground,
        )
        # InputBuffer가 map-to-base pose를 연결했으므로 sweep base/ego 좌표를
        # 최신 anchor의 base/ego 좌표로 변환한다.
        matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        aligned = _transform_xyz(points_base, matrix)
        merged.append(aligned)

    points = np.concatenate(merged, axis=0)
    # ROS x-forward/y-left를 pinned CARLA x-forward/y-right로 정확히 한 번 변환한다.
    points[:, 1] *= -1.0
    # lead/inference/sensor_agent.py:386-413의 quantize/raster/PNG 단계.
    for axis, precision in enumerate(point_precision_xyz):
        points[:, axis] = np.round(points[:, axis] / precision) * precision
    raster = rasterize_lidar(config=training_config, lidar=points)[..., None]
    compressed = compress_float_image(raster, training_config)
    return np.asarray(decompress_float_image(compressed), dtype=np.float32).squeeze()


class _LeadOpenLoopRuntime:
    """pinned LEAD import·CUDA model·원본 LiDAR helper를 캡슐화한다.

    입력: TFv6 profile과 project root.
    반환: 외부 wrapper가 source-native forward에 사용할 runtime adapter.
    """

    def __init__(self, profile: dict[str, Any], project_root: Path) -> None:
        """원본 LEAD symbols, config, CUDA ensemble, LiDAR helper를 load한다.

        입력: 검증을 통과한 TFv6 profile과 workspace root.
        부작용: source checkout import path를 앞에 두고 CUDA model을 초기화한다.
        """
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
        """pose-align된 raw LiDAR history를 CUDA batch raster tensor로 바꾼다.

        입력: InputBuffer가 만든 ``AlignedPointCloudSweep`` 10개.
        반환: shape ``[1, 1, H, W]``의 source input tensor.
        """
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
        """수치 값을 CUDA float32 tensor로 옮긴다.

        입력: tensor 또는 numpy/sequence 값.
        반환: CUDA device의 float32 tensor.
        """
        if self._torch.is_tensor(value):
            return value.to(device="cuda", dtype=self._torch.float32)
        return self._torch.as_tensor(value, dtype=self._torch.float32, device="cuda")

    def image_tensor(self, value: Any) -> Any:
        """ROS image를 RGB CHW CUDA batch tensor로 바꾼다.

        입력: ``sensor_msgs/Image`` 또는 source-compatible image.
        반환: batch 차원이 있는 image tensor.
        """
        tensor = self.tensor(ros_image_to_chw(value))
        return tensor.unsqueeze(0) if tensor.ndim == 3 else tensor

    def batch_vector(self, value: Any) -> Any:
        """1차원 model feature에 필요할 때만 batch 차원을 추가한다.

        입력: 수치 vector 또는 tensor.
        반환: ``[1, N]`` 또는 기존 batch를 보존한 CUDA tensor.
        """
        tensor = self.tensor(value)
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

    def forward(self, data: dict[str, Any]) -> Any:
        """원본 LEAD OpenLoopInference의 data dict forward를 호출한다.

        입력: rgb, rasterized_lidar, command, target point를 가진 source data.
        반환: ``pred_future_waypoints``를 포함해야 하는 원본 output.
        """
        return self._model.forward(data)


class TFv6Model:
    """pinned LEAD open-loop ensemble을 ``ModelPrediction`` 계약으로 감싼다.

    입력: TFv6 selected profile, project root, 선택적 test runtime.
    반환: engine이 공통으로 소비하는 8개 CARLA-relative waypoint prediction.
    """

    def __init__(
        self,
        selected_profile: Any,
        project_root: str | Path,
        source_runtime: Any = None,
    ) -> None:
        """lazy-load source runtime을 갖는 TFv6 wrapper를 초기화한다.

        입력: 선택 profile, project root, 선택적으로 주입된 source runtime.
        부작용: 실제 CUDA load는 ``load`` 또는 첫 ``forward``까지 지연한다.
        """
        self.selected_profile = selected_profile
        self.project_root = Path(project_root).resolve()
        self._source_runtime = source_runtime

    def load(self) -> "TFv6Model":
        """asset hash를 검증하고 아직 없으면 pinned LEAD runtime을 load한다.

        입력: 없음.
        반환: load된 자기 자신.
        부작용: source/checkpoint 검증과 CUDA model 초기화를 수행할 수 있다.
        """
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
        """TFv6 frame을 원본 data dict로 변환해 8개 waypoint를 추론한다.

        입력: panorama·10 LiDAR sweep·status·세 target이 든 causal frame, 미사용 seed.
        반환: ``pred_future_waypoints``를 담은 ``ModelPrediction``.
        """
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
            """test runtime도 지원하도록 선택적으로 tensor 변환을 적용한다.

            입력: source data dict에 넣을 수치 값.
            반환: runtime converter가 있으면 변환한 값, 없으면 원래 값.
            """
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
