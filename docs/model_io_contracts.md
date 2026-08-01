# Model I/O Contracts

> Source-derived runtime contract for TFv6, InterFuser, DiffusionDrive V2, and SparseDrive V2.  
> Companion architecture document: [Inference Layer 설계](inference_engine_design.md).

## 1. Scope and notation

This document describes the tensor/data contract between the external PreprocessLayer, the inference-layer OptFeederNode/InputBuffer, and each concrete PyTorch model object. It does not define a generic adapter.

The Catkin source layout is a package group at `src/Validator_pkg/`: `validator_inference`, `validator_msgs`, and `validator_pipeline` are sibling packages. The last directory keeps the ROS package name `Validator_pkg`, so the integrated entry point is `roslaunch Validator_pkg openloop_validator.launch`.

- B: batch size; open-loop runtime uses B=1.
- All ROS model output paths (`/waypoints`) use frame Lidar3D-4.
- Visualization uses a base_link BEV (600×500, x=[0,25] m, y=[-15,15] m, 0.05 m/pixel) and applies the recorded T_base_from_lidar to Lidar3D-4 waypoints before drawing them.
- InputBuffer chooses only messages at or before DrivingFeature.header.stamp. Stale messages are valid and their age is recorded.
- Static model binary assets are local files under models/ and are verified before model load.
- The current bag has no radar and no camera intrinsic/distortion data.

## 2. Source revisions and local assets

| Model | Source revision | Required local assets |
| --- | --- | --- |
| TFv6 | autonomousvision/lead `730bc1a2f44d5f28312dd55f0ca958e94a24c038` (verified checkpoint-compatible public `cvpr2026` pin); checkpoint ln2697/tfv6 `noradar_resnet34` | config.json; model_0030_1.pth; model_0030_2.pth |
| InterFuser | opendilab/InterFuser f0be8ea294c49a59febaa032dd6da40b475e3657 | official_carla/interfuser.pth.tar |
| DiffusionDrive V2 | hustvl/DiffusionDriveV2 1cd12a1e155c34dcc471261835444c8d5587580b | diffusiondrivev2_sel.ckpt; kmeans_navsim_traj_20.npy; ResNet-34 backbone |
| SparseDrive V2 | swc-17/SparseDriveV2 696ef77924eb9e0a4b4047d013a50e9854bfa026 | checkpoint; path/velocity/trajectory vocabularies; ResNet-34 backbone |

The DiffusionDrive V2 anchor is an official repository binary, not a user-generated input.

~~~text
models/
  diffusiondrive_v2/navsim_v2/
    diffusiondrivev2_sel.ckpt
    kmeans_navsim_traj_20.npy
~~~

The anchor shape is (20, 8, 2), dtype float64 in the source file, and the model constructor converts it to a non-trainable float32 tensor. Its verified SHA-256 is 2d341e525d7882ee5e916c67d37bea571e4c793cf801f0fff3eff59d38b20d98.

## 3. TFv6

### 3.1 Inputs

| Input | Runtime representation | Provider |
| --- | --- | --- |
| RGB | source camera order left, front, right; transport image is RGB 384×1152 and is split into three 384×384 views before native forward | PreprocessLayer |
| LiDAR | ten chronological raw sweeps become B×1×320×384 raster | InputBuffer FIFO plus TFv6 source builder |
| command/speed | CARLA command one-hot [LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGELANELEFT, CHANGELANERIGHT] plus forward speed m/s | OptFeederNode |
| route | previous/current/next ego-local points | OptFeederNode |

The LiDAR raster uses x=[-32,64] m, y=[-40,40] m, 4 pixels/m, occupancy clip=5. InputBuffer pairs each of the ten chronological PointCloud2 messages with the interpolated map-to-base pose transform into the newest causal anchor; a missing pose is `not_ready`. For every sweep, the TFv6 source boundary first applies the profile's full `T_base_from_lidar`, then ports the pinned LEAD `base_agent.py:140-237` order exactly: ego-footprint mask, `lead.common.ransac.remove_ground(..., parallel=True)`, and source BEV/height bounds. Only the surviving base/ego points are aligned into the newest anchor. ROS y-left is converted once to CARLA y-right immediately before the pinned 0.1 m quantization, LEAD `rasterize_lidar`, and LEAD 16-bit PNG `compress_float_image`/`decompress_float_image` round-trip.

The inference-layer subscriptions for this path are raw `/lidar3D` and `/tf`; `/lidar/tfv6/processed` is an optional five-sweep preprocessing diagnostic and is never a TFv6 model input. Pose lookup uses linear translation interpolation and quaternion SLERP between bracketing `/tf` messages for each of the ten raw sweep stamps.

Fallback route points remain published in `Lidar3D-4`. Before TFv6 forward they are transformed by the recorded `T_base_from_lidar` (`base_link` parent, `Lidar3D-4` child, translation `[1.5,0,1.25]`, quaternion `[0,0,0,1]`) and only then converted to CARLA y-right coordinates.

### 3.2 Official no-radar contract

The selected official `noradar_resnet34/config.json` sets `use_radar_detection: false`. At public source pin `730bc1a2f44d5f28312dd55f0ca958e94a24c038`, `OpenLoopInference` loads all `model*.pth` members, so the official two-member ensemble is `model_0030_1.pth` and `model_0030_2.pth`. The no-radar path builds and reads no radar feature. OptFeederNode therefore leaves `DrivingFeature.tfv6_radar` empty with rows=cols=0 and records no synthetic-radar compatibility flag.

This source SHA is the exact verified compatible public `cvpr2026` pin. The Hugging Face config contains no provenance SHA for a private training commit, so this document does not claim that identity. Runtime asset SHA-256 values are computed and recorded after download rather than invented in the profile.

### 3.3 Output

Native prediction is eight ego-relative waypoint points at 0.25 s intervals. First map CARLA to ROS base coordinates with x_ros=x_carla and y_ros=-y_carla; then convert base coordinates to Lidar3D-4 coordinates as specified in the architecture document.

## 4. InterFuser

### 4.1 Inputs

| Input | Tensor | Provider |
| --- | --- | --- |
| front image | B×3×224×224 RGB | PreprocessLayer |
| left image | B×3×128×128 RGB | PreprocessLayer |
| right image | B×3×128×128 RGB | PreprocessLayer |
| front-center image | B×3×128×128 RGB crop of front, not a fourth camera | PreprocessLayer |
| LiDAR histogram | B×3×224×224 float32 channels [below, above, total] | PreprocessLayer |
| status | B×7: CARLA command[6] and forward speed | OptFeederNode |
| target point | B×2 CARLA ego-local route point | OptFeederNode |

Image tensors use the official resize/crop and ImageNet normalization. The pinned LiDAR builder uses the exact edge arrays `xbins=np.linspace(-28,29,225)` and `ybins=np.linspace(-28,0,225)`: 225 edge values and therefore 224×224 bins. It then clips each height channel at 5. The asymmetric x endpoint is intentional and must not be normalized to 28. The `Lidar3D-4` target point is transformed by the same recorded `T_base_from_lidar` before the CARLA y sign change. InterFuser requires no temporal FIFO and outputs ten ego-relative waypoints.

### 4.2 Current-bag compatibility

The current hardware camera resolutions/FOV differ from CARLA training cameras. Deterministic conversion makes the tensor contract executable but every run records the camera-rig compatibility flag. It is an open-loop comparison condition, not a training-domain equivalence claim.

## 5. DiffusionDrive V2

### 5.1 Inputs

| Input | Tensor | Provider |
| --- | --- | --- |
| stitched camera | B×3×256×1024 RGB in [0,1] | PreprocessLayer |
| LiDAR histogram | B×1×256×256 float32, current sweep only | PreprocessLayer |
| status | B×8: NAVSIM command[4], ego velocity[2], ego acceleration[2] | OptFeederNode |
| plan anchor | 20×8×2 static tensor | local official asset |

NAVSIM command order is [LEFT, STRAIGHT, RIGHT, UNKNOWN]. The fallback is STRAIGHT. LiDAR is z>0.2 histogram with x/y=[-32,32] m, 4 pixels/m, clip=5, z<100. No history FIFO is required.

### 5.2 Candidate selection and output

The original SEL inference path performs these neural operations.

1. Repeat 20 official anchors across ten groups and add truncated diffusion noise, producing 200 candidate starts.
2. Run source diffusion denoising, yielding B×200×8×3 candidate trajectories.
3. The checkpoint's coarse scorer predicts no-collision, drivable-area, ego-progress, TTC, and comfort proxies; retain the top 32 candidates.
4. The checkpoint's three-stage fine scorer selects one candidate at each stage.
5. Export the final selected candidate: traj_to_score[:, -1], shape B×8×3.

PDM is excluded. The inference object neither imports nor initializes the PDM scorer/process pool, reads a metric cache, or returns reward_dict. That code belongs to the repository's evaluation branch, not to this project's waypoint inference path. The seed is deterministically derived from model ID, run ID, and reference stamp and is recorded.

The official checkpoint contains 972 `agent._transfuser_model.*` entries. The generated PDM-free derivative intentionally has no evaluation-only `_trajectory_head.vocab`; loading removes only that named entry and requires an exact strict match for all remaining 971 entries. Any other missing or unexpected key is a load error with model/checkpoint context.

The output has eight (x,y,heading) poses over 4.0 s at 0.5 s intervals in NAVSIM ego coordinates.

## 6. SparseDrive V2

### 6.1 Inputs

| Input | Tensor/data | Provider |
| --- | --- | --- |
| images | B×3 cameras×3×256×512 RGB, source normalization | PreprocessLayer |
| status | B×8 NAVSIM status | OptFeederNode |
| projection conditions | per-camera lidar2img and image dimensions | OptFeederNode from profile inline calibration |
| raw LiDAR | not a model input | none |

Camera order is left, front, right. The source converts each image through 512×288 resize, crop (0,32,512,288), final 256×512, and mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375] normalization.

SparseDrive's projection matrices are model conditions. They are not concatenated into image channels.

### 6.2 Inline calibration schema

Only the selected SparseDrive profile carries calibration. The user fills each camera's K_original_3x3, distortion model/coefficients, and T_lidar_from_camera_4x4.

The pinned SparseDrive inference builder keeps distortion model/coefficients as calibration metadata but does not undistort the selected camera images or consume those coefficients. Projection uses the raw-image intrinsic composed with the source 1920x1080-equivalent -> 512x288 -> crop `(0,32,512,256)` geometry.

~~~text
T_camera_from_lidar = inverse(T_lidar_from_camera)
lidar2img = A_resize_crop × K_original × T_camera_from_lidar[:3, :]
~~~

Before forwarding, OptFeederNode validates that all three cameras are complete and hashes the canonical inline calibration object into calibration_profile_sha256. Selecting SparseDrive with missing values raises a profile error before worker construction; the incomplete SparseDrive profile does not block the other three models.

### 6.3 Output

The model returns B×8×3 NAVSIM ego SE2 poses over 4.0 s at 0.5 s intervals. NAVSIM x-forward/y-left is already ROS base orientation, then the common base-to-Lidar3D-4 transform is applied.

## 7. Runtime readiness matrix

| Model | Current bag | Extra user data | Compatibility state |
| --- | --- | --- | --- |
| TFv6 | image/LiDAR/ego/TF available; official selected checkpoint requires no radar | none | executable after official config/two weights and 10-sweep warm-up |
| InterFuser | required dynamic inputs available after preprocessing | none | executable after checkpoint download |
| DiffusionDrive V2 | required dynamic inputs available after preprocessing | none | executable after official assets download |
| SparseDrive V2 | images and ego available; calibration absent | inline profile calibration for all three cameras | blocked until calibration is complete |

After the model assets are downloaded and the implementation passes its smoke tests, the only runtime value the user must supply is SparseDrive V2's inline calibration object.

## 8. Runtime command contract

각 모델의 source/checkpoint/auxiliary asset은 선택한 local profile을 기준으로 먼저 내려받고 검증한다. 예시는 InterFuser이며 다른 모델은 model ID를 동일하게 바꾼다.

```bash
catkin_make
source devel/setup.bash
rosrun validator_inference fetch_model_assets.py --model-id interfuser
roslaunch Validator_pkg openloop_validator.launch model_id:=interfuser show_rviz:=true
```

`openloop_validator.launch`는 선택한 model ID를 PreprocessLayer, inference layer, visualization layer에 동일하게 전달한다. 별도 terminal에서 ROS simulated time을 공급한다. Inference layer만 독립 점검할 때는 `roslaunch validator_inference openloop_inference.launch model_id:=interfuser`를 사용한다.

```bash
source devel/setup.bash
rosbag play --clock rosbag/2026-07-16-17-13-14.bag
```

launch의 `profile_file`은 로컬 JSON만 허용하며 `record_dir`은 기본적으로 프로젝트의 `records/openloop/`다. SparseDrive V2는 profile의 세 calibration record가 완성되기 전에는 worker를 만들지 않는다.

## 9. References

- [TFv6 model card](https://huggingface.co/ln2697/tfv6)
- [LEAD source](https://github.com/autonomousvision/lead)
- [InterFuser source](https://github.com/opendilab/InterFuser)
- [DiffusionDrive V2 source](https://github.com/hustvl/DiffusionDriveV2)
- [SparseDrive V2 source](https://github.com/swc-17/SparseDriveV2)
