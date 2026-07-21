# Inference Layer 설계

> 상태: 구현 전 확정 설계 v0.7  
> 범위: InferenceEngineNode, InputBuffer, InferenceWorker, OptFeederNode, RecorderNode와 TFv6 / InterFuser / DiffusionDrive V2 / SparseDrive V2의 open-loop inference 계약. Perception을 만드는 PreprocessLayer와 결과를 표시하는 VisualizationLayer는 엔진 밖 layer로만 표기한다.

모델별 실제 tensor·asset 계약은 [Model I/O Contracts](model_io_contracts.md)에 분리한다. 이 문서는 ROS 객체, 시간·좌표·기록·readiness 계약을 고정한다.

## 1. 확정 원칙

- ~model_id가 고른 하나의 profile만 로컬 [profile.json](../src/Validator_pkg/config/profile.json)에서 읽는다. 원격 profile 요청은 없다.
- ModelAdapter는 없다. TFv6Model, InterFuserModel, DiffusionDriveV2InferenceModel, SparseDriveV2Model은 각각 원본 checkpoint의 입력 순서와 forward만 직접 가진다. 공통 wrapper가 모델 feature를 변환하지 않는다.
- InputBuffer가 inference layer의 모든 subscriber를 소유한다. callback은 lock 아래 timestamped message만 보관한다. 다수 relay/subscriber node를 만들지 않는다.
- per-frame image/LiDAR 변환은 외부 PreprocessLayer가 발행한다. 단, TFv6의 10 raw LiDAR sweep은 InputBuffer FIFO와 TFv6Model source temporal builder가 함께 처리한다. 각 raw sweep에는 먼저 profile의 full T_base_from_lidar와 pinned LEAD ego mask/RANSAC/bounds를 적용하고, 그 뒤 reference stamp 기준 motion compensation을 한다.
- main loop는 bag time 기준 rospy.Timer 10 Hz다. worker queue는 없고 forward는 하나만 실행한다. busy tick은 기다리지 않고 busy_tick_skipped로 기록한다.
- stale input은 inference를 막지 않는다. 하지만 모든 model input은 reference_stamp 이하의 가장 최신 sample만 사용한다. 미래 sample은 절대 사용하지 않는다.
- reference stamp는 선택된 /driving_feature.header.stamp다. non-history input은 현재·직전 두 sample을 보관해 causal selection을 보장한다. TFv6 raw LiDAR는 10개 FIFO를 보관한다.
- 별도 /inference_time topic은 없다. /inference_metric의 validator_msgs/InferenceMetric.elapsed_ms가 성공 forward의 milliseconds 정본이며, forward하지 않은 not_ready와 busy_tick_skipped는 -1.0이다. CSV 행은 /inference_metric 하나당 정확히 한 행이다.
- RecorderNode는 시작 즉시 <project-root>/records/openloop/에 inference_{YYYYMMDDTHHMMSS+0900}_{model_id}.csv를 연다. bag time은 tick/stamp에만, system wall clock은 파일명·run 시작 시각·tick wall time에만 쓴다.
- 모든 published path와 temporary route point frame은 실제 bag LiDAR frame Lidar3D-4다.

## 2. 입력 ownership과 시간 계약

| 소유 객체/layer | 책임 | 하지 않는 일 |
| --- | --- | --- |
| PreprocessLayer | profile별 single-frame image resize/crop/stitch와 LiDAR histogram/BEV tensor 발행 | engine history·CSV·worker 관리 |
| OptFeederNode | /Ego_topic에서 model-native status, command fallback, route fallback, SparseDrive inline calibration을 /driving_feature에 발행. TFv6 optional radar field는 비운다 | image/LiDAR subscriber 보유 |
| InputBuffer | 모든 input subscriber, timestamp history, TFv6 10-sweep FIFO, causal snapshot | feature를 다른 모델 형식으로 재가공 |
| 선택 모델 객체 | 원본 checkpoint load와 native forward; TFv6만 source temporal LiDAR builder | topic subscribe/publish/CSV |
| RecorderNode | metric event를 CSV 한 행으로 기록 | inference 결과 선택/재계산 |

InputBuffer.snapshot_at(reference_stamp)는 다음과 같이 동작한다.

1. 각 required key에서 header.stamp <= reference_stamp인 sample 중 가장 최신을 선택한다.
2. 한 번도 받은 적이 없거나 조건을 만족하는 sample이 없으면 not_ready다. stale이라서 not_ready가 되지는 않는다.
3. age는 reference_stamp - input_stamp milliseconds다. causal rule이므로 성공 frame의 age는 음수가 될 수 없다.
4. TFv6는 reference stamp 이하의 최신 LiDAR sweep을 anchor로 잡고, 그 이전 9개와 각 sweep stamp의 map -> base_link pose를 사용한다. 10개가 모두 없거나 pose 보간 bracket gap이 120 ms보다 크면 not_ready다.

## 3. 현재 rosbag 사실

| 항목 | 계약 |
| --- | --- |
| raw LiDAR | /lidar3D, sensor_msgs/PointCloud2, frame Lidar3D-4, x,y,z,intensity float32 |
| LiDAR rate | median 107 ms, sweep당 약 28,800 points |
| camera | front Camera-1 JPEG 1280×720, left Camera-2 JPEG 640×480, right Camera-3 JPEG 640×480 |
| ego | /Ego_topic, morai_msgs/EgoVehicleStatus |
| TF | dynamic /map -> /base_link, static /base_link -> Lidar3D-4 |
| 없는 data | CameraInfo, intrinsic, distortion, radar topic |

## 4. 시퀀스 다이어그램

엔진 내부 participant는 실제 객체다. 엔진 밖은 layer로 합쳤다. ROS 표준 publisher는 participant로 만들지 않고 publish(/topic) 호출에만 표시했다. InputBuffer 내부 subscriber callback self-loop는 의도적으로 생략했다.

~~~mermaid
sequenceDiagram
    autonumber
    participant RS as ReplaySourceLayer
    participant PL as PreprocessLayer
    participant OF as OptFeederNode
    participant N as InferenceEngineNode
    participant S as LocalProfileStore
    participant B as InputBuffer
    participant T as MainLoopTimer
    participant W as InferenceWorker
    participant M as SelectedModel
    participant PB as PathMessageBuilder
    participant R as RecorderNode
    participant CW as CsvRunWriter
    participant V as VisualizationLayer

    R->>CW: open(record_dir, wall_time, model_id)
    CW-->>R: CSV ready
    N->>S: load_local(model_id)
    S-->>N: ModelProfile
    N->>B: configure(profile subscriptions)
    N->>W: load(profile runtime assets)
    W->>M: load checkpoint and source code
    M-->>W: ready or asset error
    OF->>S: load_local(model_id)
    S-->>OF: ModelProfile with inline static inputs

    RS-->>OF: publish(/Ego_topic)
    OF-->>B: publish(/driving_feature)
    OF-->>B: publish(/route/target_point*) when required
    PL-->>B: publish(profile perception tensors)
    RS-->>B: publish(/lidar3D and /tf for TFv6 only)

    T->>N: on_timer_tick()
    N->>W: take_completed()
    alt completion mailbox has result
        W-->>N: prediction and elapsed_ms
        N->>PB: build(prediction, reference_stamp, Lidar3D-4)
        PB-->>N: nav_msgs/Path
        N-->>V: publish(/waypoints)
        N-->>R: publish(/inference_metric, inference_completed, elapsed_ms)
    end
    N->>B: snapshot_at(reference_stamp)
    alt required causal inputs and runtime assets exist
        B-->>N: InferenceFrame
        N->>W: try_submit(frame)
        alt worker idle
            W-->>N: accepted
            W->>M: forward(frame)
            M-->>W: prediction and elapsed_ms
            W-->>W: store one completion result
        else worker busy
            W-->>N: rejected_busy
            N-->>R: publish(/inference_metric, busy_tick_skipped, elapsed_ms=-1.0)
        end
    else input or asset absent
        B-->>N: NotReady(missing keys)
        N-->>R: publish(/inference_metric, not_ready, elapsed_ms=-1.0)
    end

    R->>CW: append_row(metric)
~~~

Recorder는 /inference_metric만 구독한다. elapsed_ms는 inference_completed일 때만 실제 forward 시간(ms)이고, forward하지 않은 outcome은 -1.0으로 명시한다.

## 5. 클래스 다이어그램

~~~mermaid
classDiagram
    direction LR

    class InferenceEngineNode {
        -InputBuffer input_buffer
        -InferenceWorker inference_worker
        -PathMessageBuilder path_builder
        +start() void
        +on_timer_tick() void
        +publish_completed(result) void
        +publish_metric(metric) void
    }

    class InputBuffer {
        -Mutex lock
        -Map current_and_previous
        -Map timed_fifos
        -TfPoseHistory tf_pose_history
        +configure(profile) void
        +on_message(key, message) void
        +on_tf(message) void
        +snapshot_at(reference_stamp) FrameOrNotReady
    }

    class TimedFifo {
        +int capacity
        +append(stamped_message) void
        +latest_at_or_before(stamp) MessageOrMissing
        +latest_n_at_or_before(count, stamp) List
    }

    class TfPoseHistory {
        +interpolate_map_to_base(stamp, max_gap_ms) PoseOrMissing
    }

    class InferenceFrame {
        +Time reference_stamp
        +Map inputs
        +Map input_age_ms
        +int history_count
        +float history_span_ms
    }

    class InferenceWorker {
        -Mutex forward_lock
        -SelectedModel selected_model
        -CompletionMailbox completion_mailbox
        +load(profile) void
        +try_submit(frame) SubmitResult
        +take_completed() ResultOrEmpty
    }

    class CompletionMailbox {
        +store_once(result) void
        +take() ResultOrEmpty
    }

    class TFv6Model {
        +forward(frame) ModelPrediction
    }
    class InterFuserModel {
        +forward(frame) ModelPrediction
    }
    class DiffusionDriveV2InferenceModel {
        +forward(frame, rng_seed) ModelPrediction
    }
    class SparseDriveV2Model {
        +forward(frame) ModelPrediction
    }

    class OptFeederNode {
        -LocalProfileStore profile_store
        +on_ego_status(message) void
        +build_status_feature(ego) DrivingFeature
        +build_fallback_route_points() PointStamped[]
        +validate_inline_calibration() void
    }

    class DrivingFeature {
        +Header header
        +string model_id
        +float32[] status_feature
        +float32[] tfv6_radar
        +CameraCalibration[] camera_calibrations
        +string source
    }

    class InferenceMetric {
        +string run_id
        +uint64 tick_index
        +string outcome
        +float64 elapsed_ms
        +string[] compatibility_flags
        +uint64 rng_seed
    }

    class RecorderNode {
        -CsvRunWriter writer
        +on_metric(metric) void
    }

    InferenceEngineNode *-- InputBuffer
    InferenceEngineNode *-- InferenceWorker
    InferenceWorker *-- CompletionMailbox
    InputBuffer *-- TimedFifo
    InputBuffer *-- TfPoseHistory
    InputBuffer --> InferenceFrame
    InferenceWorker ..> TFv6Model : selected by model_id
    InferenceWorker ..> InterFuserModel : selected by model_id
    InferenceWorker ..> DiffusionDriveV2InferenceModel : selected by model_id
    InferenceWorker ..> SparseDriveV2Model : selected by model_id
    OptFeederNode ..> DrivingFeature : publishes
    InferenceEngineNode ..> InferenceMetric : publishes
    RecorderNode --> InferenceMetric : consumes
~~~

SelectedModel은 runtime에 선택된 concrete object를 뜻할 뿐 interface/adapter class가 아니다. 네 concrete model class 중 하나만 worker에 넣는다.

## 6. 확정된 P0 계약

### 6.1 TFv6 history와 official no-radar checkpoint

- first 10 LiDAR sweep 구간은 not_ready이며 CSV에 남긴다. 임의 history padding을 하지 않는다.
- 이후 history의 stale sweep은 사용하고 age/history span을 기록한다.
- 각 chronological PointCloud2는 해당 stamp의 map -> base_link pose와 결합되고, newest causal sweep 기준 `T_anchor_lidar_from_sweep_lidar`를 가진다. pose가 없거나 120 ms bracket 제한을 넘으면 해당 tick은 not_ready다.
- model boundary는 ROS-native PointCloud2 reader로 finite xyz를 decode하고 sweep을 anchor에 정렬한다. 그 뒤 CARLA y-right 변환, LEAD source의 0.1 m x/y/z quantization, `rasterize_lidar`, 16-bit PNG `compress_float_image`/`decompress_float_image` 순서를 그대로 적용한다.
- `model_id=tfv6`는 공식 Hugging Face `noradar_resnet34` config와 `model_0030_1.pth`, `model_0030_2.pth` ensemble을 사용한다. config의 `use_radar_detection=false` 경로에서는 pinned `OpenLoopInference`가 radar feature를 만들거나 읽지 않는다.
- 소스는 public `cvpr2026` branch의 검증된 checkpoint-compatible pin `730bc1a2f44d5f28312dd55f0ca958e94a24c038`이다. Hugging Face config가 private training commit SHA까지 제공하는 것은 아니므로 그 provenance를 주장하지 않는다.
- OptFeederNode는 `DrivingFeature.tfv6_radar`를 빈 vector, rows=0, cols=0으로 발행한다. synthetic radar와 compatibility flag는 생성하지 않는다. ROS message의 optional radar fields는 향후 checkpoint 호환성을 위해 유지한다.

### 6.2 DiffusionDrive V2는 PDM을 쓰지 않는다

공식 kmeans_navsim_traj_20.npy (20,8,2)는 checkpoint와 같은 models/diffusiondrive_v2/navsim_v2/에 둔다. DiffusionDriveV2InferenceModel은 source의 diffusion candidate generation, coarse top-32, three-stage learned fine selection을 그대로 사용하고 마지막 traj_to_score[:, -1] (8,3)만 반환한다.

공식 checkpoint의 `agent._transfuser_model.*` 972개 중 PDM 제거 derivative에 의도적으로 없는 `_trajectory_head.vocab` 한 항목만 제거한 뒤, 나머지 971개는 `strict=True`로 완전 일치해야 한다. 그 외 missing/unexpected key는 허용하지 않는다.

PDM import, PDM process pool, metric-cache, evaluator, reward_dict return은 inference class에 없다. 매 frame RNG seed는 sha256(model_id + reference_stamp_nsec + run_id)를 uint64로 줄인 값이며 metric/CSV에 기록한다.

### 6.3 SparseDrive V2 calibration은 profile 내부 값이다

별도 calibration file은 없다. profile.json의 sparsedrive_v2.static_runtime_inputs.camera_calibration 안에서 user가 아래 값을 세 camera 모두 채운다.

~~~text
K_original_3x3
distortion_model
distortion_coefficients
T_lidar_from_camera_4x4
~~~

runtime은 T_camera_from_lidar = inverse(T_lidar_from_camera)를 만들고, 원본 source resize/crop matrix A_resize_crop를 적용해 lidar2img = A_resize_crop * K_original * T_camera_from_lidar[:3,:]를 구성한다. 고정 source geometry는 raw image를 1920x1080-equivalent로 놓은 뒤 512x288로 resize하고 (x=0, y=32, w=512, h=256)으로 crop한다. 이 pinned source path는 distortion metadata를 보관하지만 image undistortion이나 계수 적용을 수행하지 않는다. calibration이 하나라도 비어 있으면 SparseDrive만 not_ready다.

### 6.4 output frame 변환

CARLA model output은 x_ros=x_carla, y_ros=-y_carla로 먼저 base_link 좌표가 된다. NAVSIM output은 x-forward/y-left라 부호 변경 없이 base_link 좌표가 된다. bag TF가 T_base_from_lidar이므로 Path pose에는 아래를 적용한다.

~~~text
p_lidar = inverse(T_base_from_lidar) * p_base
~~~

이후 Path.header.frame_id와 모든 PoseStamped.header.frame_id는 Lidar3D-4다.

## 7. OptFeeder contract

| 모델 | /driving_feature 내용 | route fallback |
| --- | --- | --- |
| TFv6 | CARLA command[6], forward speed; radar vector empty, rows=cols=0 | LANEFOLLOW; previous/current/next = 5/15/30 m forward |
| InterFuser | CARLA command[6], forward speed | LANEFOLLOW; 15 m forward |
| DiffusionDrive V2 | NAVSIM command[4], ego velocity[2], ego acceleration[2] | STRAIGHT |
| SparseDrive V2 | NAVSIM status[8], validated inline calibration | STRAIGHT |

모든 temporary route point는 Lidar3D-4 frame에서 발행한다. status feature의 source는 fallback으로 명시한다.

CARLA 계열 TFv6/InterFuser에 target을 넣을 때는 profile에 기록된 `T_base_from_lidar`를 먼저 적용한다. 이 bag의 고정 transform은 parent=base_link, child=Lidar3D-4, translation=[2,0,1.5] m, quaternion=[0,0,0,1]이다. 따라서 5/15/30 m fallback은 회전이 identity인 현재 bag에서 base x=7/17/32 m가 된 다음 CARLA y-right 부호 변환을 거친다.

## 8. ROS message와 code-level 경계

다이어그램만으로는 wire-format, lifecycle, test target이 고정되지 않는다. 구현은 아래 message와 module boundary를 그대로 따른다.

~~~text
CameraCalibration.msg
string name
string frame_id
uint32 image_width
uint32 image_height
float64[9] k_original
string distortion_model
float64[] distortion_coefficients
float64[16] t_lidar_from_camera

DrivingFeature.msg
std_msgs/Header header
string model_id
float32[] status_feature
float32[] tfv6_radar
uint32 tfv6_radar_rows
uint32 tfv6_radar_cols
CameraCalibration[] camera_calibrations
string source

InferenceMetric.msg
std_msgs/Header header
string run_id
uint64 tick_index
string model_id
string outcome
float64 elapsed_ms
string[] missing_input_keys
string[] compatibility_flags
string[] input_keys
float64[] input_age_ms
uint32 history_count
float64 history_span_ms
uint64 rng_seed
string profile_sha256
string calibration_profile_sha256
~~~

~~~text
src/Validator_pkg/
  config/profile.json
  validator_msgs/
    msg/CameraCalibration.msg
    msg/DrivingFeature.msg
    msg/InferenceMetric.msg
  validator_inference/
    package.xml
    scripts/opt_feeder_node.py
    scripts/inference_engine_node.py
    scripts/recorder_node.py
    src/validator_inference/profile_store.py
    src/validator_inference/input_buffer.py
    src/validator_inference/worker.py
    src/validator_inference/path_builder.py
    src/validator_inference/models/tfv6.py
    src/validator_inference/models/interfuser.py
    src/validator_inference/models/diffusiondrive_v2.py
    src/validator_inference/models/sparsedrive_v2.py
    launch/openloop_inference.launch
    test/
~~~

DiagnosticArray는 fixed CSV schema가 아니므로 쓰지 않는다. InferenceMetric만 row event다.

## 9. open-loop readiness

| 모델 | 결정된 실행 경로 | user가 채울 값 | 구현 후 실행 조건 |
| --- | --- | --- | --- |
| TFv6 | official `noradar_resnet34` two-weight ensemble + 10-sweep | 없음 | config/weights, 10 sweep warm-up, paired TF pose |
| InterFuser | official single-frame forward | 없음 | checkpoint |
| DiffusionDrive V2 | official anchor + PDM-free source inference path | 없음 | checkpoint, anchor, backbone |
| SparseDrive V2 | official calibration-conditioned forward | inline camera calibration | checkpoint assets와 세 camera calibration 완성 |

따라서 이 설계를 구현·smoke test한 뒤 사용자가 직접 넣는 runtime 정보는 SparseDrive V2 profile 내부 calibration 값뿐이다. checkpoint, backbone, vocabulary, official anchor, PDM 제거, TFv6 official no-radar 경로, route fallback, shape/coordinate/timing 계약은 구현 책임이다.

이 문서의 범위는 rosbag open-loop evaluator다. /waypoints만 발행하며 controller나 safety supervisor를 포함하지 않으므로 실차 주행 배포 설계는 아니다.

## 10. 구현 검증 순서

1. InputBuffer unit test: causal selection, stale acceptance, future sample exclusion, TFv6 10 FIFO warm-up, pose interpolation gap, age record.
2. worker integration test: queue depth 0, busy tick metric, one-slot completion mailbox, exactly one success metric with elapsed_ms.
3. OptFeederNode test: command order, ego-frame velocity/acceleration, route fallback frame, TFv6 empty radar fields, inline calibration validation.
4. Recorder test: node-start CSV, one row for every outcome, metric field-to-column mapping, and no secondary timing topic.
5. per-model mock forward test followed by actual checkpoint smoke inference. DiffusionDrive V2 smoke test asserts no PDM import/process pool/metric-cache access.

## 11. 재현 가능한 open-loop 실행

기본 평가는 저장소 root의 `openloop` CLI로 준비하고 실행한다. 예를 들어 InterFuser는 다음 세 명령을 사용한다.

```bash
./openloop setup interfuser
./openloop interfuser --rviz
./openloop bag
```

첫 명령은 model asset과 전용 Python 환경을 준비하고 검증한다. 두 번째 명령은 PreprocessLayer, inference layer, visualization을 함께 시작하며, 세 번째 명령은 별도 terminal에서 기본 bag과 `/clock`을 재생한다.

### 고급 직접 실행

`openloop_inference.launch`만 분리 실행하거나 raw `roslaunch`/`rosbag` 인자를 직접 제어해야 할 때는 [Open-loop 운영 절차](openloop_validator.md#고급-직접-launch)의 고급 절차를 따른다. 직접 launch는 모델 환경을 생성하거나 검증하지 않는다.

통합 launch는 `/use_sim_time=true`를 설정하며 기본 CSV 위치는 프로젝트의 `records/openloop/`다. 별도의 timing topic은 없으며 `/inference_metric`의 각 메시지가 정확히 한 CSV row가 된다. SparseDrive V2의 shipped null calibration은 model/worker 생성 전에 거부된다.
