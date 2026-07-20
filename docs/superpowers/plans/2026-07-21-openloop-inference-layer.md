# Open-loop Inference Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ROS1 inference layer that runs one selected TFv6, InterFuser, DiffusionDrive V2, or SparseDrive V2 checkpoint against a preprocessed rosbag stream and writes an outcome row for every 10 Hz tick.

**Architecture:** A `validator_msgs` package owns the two inter-node contracts. `validator_inference` keeps topic callbacks and causal histories in `InputBuffer`, feeds exactly one concrete source-native PyTorch model through a one-slot worker, and uses `RecorderNode` as the only CSV writer. `OptFeederNode` turns `/Ego_topic` and the local selected profile into all model inputs other than image/LiDAR; SparseDrive calibration stays inline in that profile.

**Tech Stack:** ROS1 Noetic, catkin, `rospy`, `tf2_ros`, `sensor_msgs`, `nav_msgs`, `morai_msgs`, Python 3, NumPy, PyTorch, pytest, rosbag.

## Global Constraints

- Work only in the inference layer. `PreprocessLayer` publishes its model-specific image/LiDAR tensors and is never reimplemented here.
- Read exactly one selected profile from local `src/Validator_pkg/config/profile.json`; never request a profile remotely.
- Do not introduce `ModelAdapter`; model factory returns one of four concrete source-native objects.
- `InputBuffer` owns every inference-layer subscriber and selects samples causally: latest `header.stamp <= reference_stamp` only.
- Main loop is a `rospy.Timer` at exactly 10 Hz. Stale inputs still run; their signed ages are recorded.
- TFv6 keeps ten raw LiDAR sweeps and the matching map-to-base poses; it is `not_ready` only until history/poses exist.
- Worker concurrency is one, queue depth is zero, and a busy tick emits `busy_tick_skipped`.
- Publish every outcome to `/inference_metric` as `validator_msgs/InferenceMetric`; `elapsed_ms` is measured milliseconds only for `inference_completed` and is `-1.0` for `not_ready` or `busy_tick_skipped`.
- `RecorderNode` creates `<project-root>/records/openloop/inference_{YYYYMMDDTHHMMSS+0900}_{model_id}.csv` immediately at node startup and writes exactly one row per metric message.
- Use `Lidar3D-4` for every `nav_msgs/Path` and temporary target-point header. Convert model output through `T_base_from_lidar` exactly as declared in the profile.
- `model_id=tfv6` uses official `noradar_resnet34` config with the two `model_0030_1/2.pth` ensemble members and verified compatible public `cvpr2026` pin `730bc1a2f44d5f28312dd55f0ca958e94a24c038`; no radar feature or compatibility flag is built.
- DiffusionDrive V2 returns its source selector result `traj_to_score[:, -1]`; it imports neither PDM, metric-cache, evaluator, nor `reward_dict`.
- SparseDrive V2 starts only after its three inline calibration records are complete. The only user-provided runtime values after installation are those profile calibration numbers.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/Validator_pkg/validator_msgs/` | Compile `CameraCalibration`, `DrivingFeature`, and `InferenceMetric` ROS messages. The physical location is under `Validator_pkg`; its ROS package name remains `validator_msgs`. |
| `src/Validator_pkg/validator_inference/package.xml`, `CMakeLists.txt`, `setup.py` | Define catkin package `validator_inference` and install Python nodes. |
| `src/Validator_pkg/validator_inference/src/validator_inference/profile_store.py` | Load, validate, hash, and select the one local profile. |
| `src/Validator_pkg/validator_inference/src/validator_inference/input_buffer.py` | Own subscriber callbacks, bounded histories, causal snapshotting, and TF interpolation. |
| `src/Validator_pkg/validator_inference/src/validator_inference/opt_features.py` | Convert `EgoVehicleStatus` into command/kinematic/calibration feature values. |
| `src/Validator_pkg/validator_inference/src/validator_inference/worker.py` | One-forward, no-queue worker and one-slot completion mailbox. |
| `src/Validator_pkg/validator_inference/src/validator_inference/models/*.py` | Four concrete checkpoint loaders and source-native forward paths. |
| `src/Validator_pkg/validator_inference/src/validator_inference/path_builder.py` | Convert selected-model trajectory to `Lidar3D-4` `nav_msgs/Path`. |
| `src/Validator_pkg/validator_inference/scripts/*.py` | ROS-node composition, asset bootstrap, and node entry points. |
| `src/Validator_pkg/validator_inference/launch/openloop_inference.launch` | Start OptFeeder, engine, and recorder with sim time. |
| `src/Validator_pkg/validator_inference/test/` | Deterministic unit and ROS integration coverage. |
| `docs/inference_engine_design.md`, `docs/model_io_contracts.md`, `README.md` | Keep the shipped implementation and run command aligned with the executable contract. |

### Task 0: Create a reproducible ROS1 Noetic Docker test environment

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `docker-compose.yml`
- Test: Docker image build and a container-side ROS smoke command

**Interfaces:**
- Consumes: repository root mounted at `/workspace/E2E-OpenLoop-ROS-Validator` and the host Docker daemon.
- Produces: a `ros1-noetic` service with ROS Noetic, catkin, Python 3, NumPy, pytest, and the build tools required by later tasks.

- [ ] **Step 1: Verify the image does not yet exist**

Run: `docker image inspect e2e-openloop-ros-validator:ros-noetic`

Expected: nonzero exit because the project image has not been built.

- [ ] **Step 2: Add the minimal image and development service**

```dockerfile
FROM ros:noetic-ros-base-focal
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-catkin-tools python3-numpy python3-pip python3-pytest ros-noetic-tf2-ros \
    python3-rospkg python3-rosdep git && \
    ln -s /usr/bin/pytest-3 /usr/local/bin/pytest && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/E2E-OpenLoop-ROS-Validator
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
```

```yaml
services:
  ros1-noetic:
    build: .
    image: e2e-openloop-ros-validator:ros-noetic
    working_dir: /workspace/E2E-OpenLoop-ROS-Validator
    volumes:
      - .:/workspace/E2E-OpenLoop-ROS-Validator
    tty: true
```

`.dockerignore` excludes `.git`, `models`, `records`, `graphify-out`, `.superpowers`, and `.worktrees` from image build context.

- [ ] **Step 3: Build and run the ROS smoke command**

Run: `docker compose build ros1-noetic && docker compose run --rm ros1-noetic bash -lc 'rosversion -d && python3 -c "import numpy; print(numpy.__version__)"'`

Expected: output includes `noetic` and a NumPy version; both Docker commands exit `0`.

- [ ] **Step 4: Commit the Docker environment**

```bash
git add Dockerfile .dockerignore docker-compose.yml
git commit -m "build: add ROS Noetic development container"
```

### Task 1: Create the compiled ROS message boundary

**Files:**
- Create: `src/Validator_pkg/validator_msgs/package.xml`
- Create: `src/Validator_pkg/validator_msgs/CMakeLists.txt`
- Create: `src/Validator_pkg/validator_msgs/msg/CameraCalibration.msg`
- Create: `src/Validator_pkg/validator_msgs/msg/DrivingFeature.msg`
- Create: `src/Validator_pkg/validator_msgs/msg/InferenceMetric.msg`
- Create: `src/CMakeLists.txt`
- Test: `src/Validator_pkg/validator_msgs/test/test_message_definitions.py`

**Interfaces:**
- Consumes: `std_msgs/Header` and the profile's `camera_calibrations`/metric contract.
- Produces: importable `validator_msgs.msg.CameraCalibration`, `DrivingFeature`, and `InferenceMetric` for all three nodes.

- [ ] **Step 1: Write the failing message-definition test**

```python
from pathlib import Path


def test_message_contracts_are_compileable_ros1_fields():
    root = Path(__file__).parents[1] / "msg"
    driving = (root / "DrivingFeature.msg").read_text()
    metric = (root / "InferenceMetric.msg").read_text()
    assert "std_msgs/Header header" in driving
    assert "CameraCalibration[] camera_calibrations" in driving
    assert "float32[] tfv6_radar" in driving
    assert "string run_id" in metric
    assert "string[] input_keys" in metric
    assert "float64[] input_age_ms" in metric
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest src/Validator_pkg/validator_msgs/test/test_message_definitions.py -q`

Expected: `FAILED` because the message files do not exist.

- [ ] **Step 3: Add the exact message schemas and catkin generation configuration**

```text
# CameraCalibration.msg
string name
string frame_id
uint32 image_width
uint32 image_height
float64[9] k_original
string distortion_model
float64[] distortion_coefficients
float64[16] t_lidar_from_camera
```

```text
# DrivingFeature.msg
std_msgs/Header header
string model_id
float32[] status_feature
float32[] tfv6_radar
uint32 tfv6_radar_rows
uint32 tfv6_radar_cols
CameraCalibration[] camera_calibrations
string source
```

```text
# InferenceMetric.msg
std_msgs/Header header
string run_id
uint64 tick_index
string model_id
string outcome
float64 elapsed_ms
string[] input_keys
float64[] input_age_ms
string[] missing_input_keys
string[] compatibility_flags
uint32 history_count
float64 history_span_ms
uint64 rng_seed
string profile_sha256
string calibration_profile_sha256
```

Configure `add_message_files` with these three names, `generate_messages(DEPENDENCIES std_msgs)`, and `catkin_package(CATKIN_DEPENDS message_runtime std_msgs)` in `src/Validator_pkg/validator_msgs/CMakeLists.txt`. Declare `message_generation` for build and `message_runtime` for execution in `package.xml`. Run `catkin_init_workspace src` to create `src/CMakeLists.txt`, which is the catkin workspace toplevel required by the build command.

- [ ] **Step 4: Re-run the structural test and build messages**

Run: `pytest src/Validator_pkg/validator_msgs/test/test_message_definitions.py -q && catkin_make --pkg validator_msgs`

Expected: pytest reports `1 passed`; catkin reports `Built target validator_msgs_generate_messages_py`.

- [ ] **Step 5: Commit the message boundary**

```bash
git add src/Validator_pkg/validator_msgs
git commit -m "feat: add inference ROS message contracts"
```

### Task 2: Bootstrap the `validator_inference` catkin package and local profile store

**Files:**
- Create: `src/Validator_pkg/validator_inference/package.xml`
- Create: `src/Validator_pkg/validator_inference/CMakeLists.txt`
- Create: `src/Validator_pkg/validator_inference/setup.py`
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/__init__.py`
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/profile_store.py`
- Create: `src/Validator_pkg/validator_inference/test/test_profile_store.py`

**Interfaces:**
- Consumes: local sibling `../config/profile.json`, `~model_id`, and optional `~profile_file` ROS parameters.
- Produces: immutable `SelectedProfile(model_id, data, profile_sha256, calibration_profile_sha256)` from `load_selected_profile`.

- [ ] **Step 1: Write failing profile-store tests**

```python
import pytest
from validator_inference.profile_store import ProfileError, load_selected_profile


def test_selected_profile_has_stable_hashes(profile_path):
    selected = load_selected_profile(profile_path, "tfv6")
    assert selected.model_id == "tfv6"
    assert len(selected.profile_sha256) == 64
    assert selected.calibration_profile_sha256 == ""


def test_sparse_profile_rejects_partial_inline_calibration(profile_path):
    with pytest.raises(ProfileError, match="K_original_3x3"):
        load_selected_profile(profile_path, "sparsedrive_v2")
```

- [ ] **Step 2: Run the tests and confirm imports fail**

Run: `docker compose run --rm ros1-noetic bash -lc 'PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_profile_store.py -q'`

Expected: `ModuleNotFoundError: No module named 'validator_inference'`.

- [ ] **Step 3: Implement package metadata and profile validation**

```python
@dataclass(frozen=True)
class SelectedProfile:
    model_id: str
    data: dict[str, Any]
    profile_sha256: str
    calibration_profile_sha256: str


def load_selected_profile(profile_path: str | Path, model_id: str) -> SelectedProfile:
    payload = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    profiles = {item["id"]: item for item in payload["profiles"]}
    if model_id not in profiles:
        raise ProfileError(f"unsupported model_id: {model_id}")
    selected = copy.deepcopy(profiles[model_id])
    calibration_hash = ""
    if model_id == "sparsedrive_v2":
        calibration = selected["static_runtime_inputs"]["camera_calibration"]
        _validate_sparse_calibration(calibration)
        calibration_hash = _sha256_canonical_json(calibration)
    return SelectedProfile(model_id, selected, _sha256_canonical_json(payload), calibration_hash)
```

`_validate_sparse_calibration` requires exactly `front`, `left`, and `right`; a length-nine `K_original_3x3`; nonempty distortion model/coefficients; and a length-sixteen `T_lidar_from_camera_4x4`. It raises the named missing field, so partial calibration is an actionable startup error. Define `validator_inference` as a Python package through `catkin_python_setup()` and declare dependencies `rospy`, `tf2_ros`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `std_msgs`, `morai_msgs`, and `validator_msgs`.

- [ ] **Step 4: Run unit tests and package configuration**

Run: `docker compose run --rm ros1-noetic bash -lc 'PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_profile_store.py -q && catkin_make --force-cmake --pkg validator_msgs validator_inference'`

Expected: all profile tests pass and catkin configures `validator_inference`.

- [ ] **Step 5: Commit package bootstrap**

```bash
git add src/Validator_pkg/validator_inference/package.xml src/Validator_pkg/validator_inference/CMakeLists.txt src/Validator_pkg/validator_inference/setup.py src/Validator_pkg/validator_inference/src src/Validator_pkg/validator_inference/test/test_profile_store.py
git commit -m "feat: bootstrap local inference profile runtime"
```

### Task 3: Implement bounded causal input storage and TF pose lookup

**Files:**
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/input_buffer.py`
- Create: `src/Validator_pkg/validator_inference/test/test_input_buffer.py`

**Interfaces:**
- Consumes: `InputSpec(key, topic, ros_type, fifo_depth)`, ROS messages with `header.stamp`, and `/tf` transforms.
- Produces: `InferenceFrame(reference_stamp, inputs, input_age_ms, history_count, history_span_ms)` or `NotReady(missing_input_keys)` via `InputBuffer.snapshot_at(reference_stamp)`.

- [ ] **Step 1: Write failing causal/history tests**

```python
def test_snapshot_uses_last_sample_not_later_than_reference():
    buffer = InputBuffer([InputSpec("camera", "/cam", "sensor_msgs/Image", 2)])
    buffer.store("camera", stamped("old", 1.0))
    buffer.store("camera", stamped("future", 3.0))
    frame = buffer.snapshot_at(time_from_sec(2.0), ["camera"])
    assert frame.inputs["camera"].value == "old"
    assert frame.input_age_ms["camera"] == 1000.0


def test_tfv6_requires_exactly_ten_past_sweeps():
    buffer = tfv6_buffer()
    for index in range(9):
        buffer.store("lidar_raw", stamped(index, float(index)))
    result = buffer.snapshot_at(time_from_sec(9.0), ["lidar_raw"])
    assert result.missing_input_keys == ["lidar_raw history(10)"]
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_input_buffer.py -q`

Expected: `ModuleNotFoundError` for `validator_inference.input_buffer`.

- [ ] **Step 3: Implement the lock-protected data model**

```python
@dataclass(frozen=True)
class InputSpec:
    key: str
    topic: str
    ros_type: str
    fifo_depth: int


class InputBuffer:
    def store(self, key: str, message: Any) -> None:
        stamp_ns = _message_stamp_ns(message)
        with self._lock:
            queue = self._samples[key]
            queue.append(StampedSample(stamp_ns, message))
            while len(queue) > self._specs[key].fifo_depth:
                queue.popleft()

    def snapshot_at(self, reference_stamp: rospy.Time, required_keys: list[str]) -> InferenceFrame | NotReady:
        reference_ns = reference_stamp.to_nsec()
        with self._lock:
            selected = {key: self._latest_at_or_before(key, reference_ns) for key in required_keys}
            missing = [key for key, value in selected.items() if value is None]
            if missing:
                return NotReady(missing)
            return self._frame_from_selected(reference_ns, selected)
```

Map profile `latest_inputs` to depth two and every `history_inputs` record to its declared FIFO depth. `snapshot_at` returns no future sample, retains stale samples, calculates `reference_ns - sample_ns` in milliseconds, and adds a ten-sweep chronological `lidar_raw_history` only for TFv6. Store map-to-base transforms separately; interpolate translation linearly and quaternion with `tf.transformations.quaternion_slerp`; reject a pose bracket wider than `max_tf_bracket_gap_ms`.

- [ ] **Step 4: Add the TF interpolation test and run all buffer tests**

```python
def test_pose_interpolation_rejects_a_gap_over_profile_limit():
    history = TfPoseHistory(max_bracket_gap_ms=120)
    history.add(transform_at(0.0))
    history.add(transform_at(0.5))
    assert history.interpolate(time_from_sec(0.25)) is None
```

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_input_buffer.py -q`

Expected: all causal, age, ten-sweep, and TF-gap tests pass.

- [ ] **Step 5: Commit the input buffer**

```bash
git add src/Validator_pkg/validator_inference/src/validator_inference/input_buffer.py src/Validator_pkg/validator_inference/test/test_input_buffer.py
git commit -m "feat: add causal inference input buffer"
```

### Task 4: Implement the OptFeeder feature contract

**Files:**
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/opt_features.py`
- Create: `src/Validator_pkg/validator_inference/scripts/opt_feeder_node.py`
- Create: `src/Validator_pkg/validator_inference/test/test_opt_features.py`

**Interfaces:**
- Consumes: `morai_msgs/EgoVehicleStatus`, selected profile, and `T_base_from_lidar` from `/tf`.
- Produces: `/driving_feature` (`validator_msgs/DrivingFeature`) and zero/one/three `geometry_msgs/PointStamped` route topics in `Lidar3D-4`.

- [ ] **Step 1: Write failing feature-vector tests**

```python
def test_tf_v6_feature_uses_lane_follow_without_radar(ego):
    feature = build_driving_feature("tfv6", ego, profile("tfv6"))
    assert feature.status_feature == [0, 0, 0, 1, 0, 0, 12.5]
    assert feature.tfv6_radar == []
    assert feature.tfv6_radar_rows == feature.tfv6_radar_cols == 0


def test_navsim_feature_is_command_velocity_acceleration(ego):
    feature = build_driving_feature("diffusiondrive_v2", ego, profile("diffusiondrive_v2"))
    assert feature.status_feature == [0, 1, 0, 0, 12.5, -1.0, 0.3, 0.0]


def test_sparse_calibration_is_copied_from_the_selected_profile(ego):
    feature = build_driving_feature("sparsedrive_v2", ego, complete_sparse_profile())
    assert [item.name for item in feature.camera_calibrations] == ["front", "left", "right"]
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_opt_features.py -q`

Expected: import failure for `validator_inference.opt_features`.

- [ ] **Step 3: Implement deterministic feature creation and node publication**

```python
def build_navsim_status(ego: EgoVehicleStatus) -> list[float]:
    return [0.0, 1.0, 0.0, 0.0,
            ego.velocity.x, ego.velocity.y,
            ego.acceleration.x, ego.acceleration.y]


def build_fallback_target(stamp: rospy.Time, x_m: float) -> PointStamped:
    point = PointStamped()
    point.header.stamp = stamp
    point.header.frame_id = "Lidar3D-4"
    point.point.x, point.point.y, point.point.z = x_m, 0.0, 0.0
    return point
```

`opt_feeder_node.py` subscribes only to `/Ego_topic`. Its callback publishes the selected profile's `DrivingFeature`; TFv6 and InterFuser publish LANEFOLLOW command, DiffusionDrive V2 and SparseDrive V2 publish STRAIGHT command, and required target topics are `/route/target_point_previous`, `/route/target_point`, `/route/target_point_next` at 5/15/30 metres. It creates a `CameraCalibration` message from each complete inline sparse record, preserving `T_lidar_from_camera` rather than concatenating values to image tensors.

- [ ] **Step 4: Re-run tests and verify installed node entry point**

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_opt_features.py -q && python3 -m py_compile src/Validator_pkg/validator_inference/scripts/opt_feeder_node.py`

Expected: all feature tests pass and compilation exits `0`.

- [ ] **Step 5: Commit OptFeeder**

```bash
git add src/Validator_pkg/validator_inference/src/validator_inference/opt_features.py src/Validator_pkg/validator_inference/scripts/opt_feeder_node.py src/Validator_pkg/validator_inference/test/test_opt_features.py
git commit -m "feat: publish required non-perception model inputs"
```

### Task 5: Add concrete checkpoint runtime and reproducible asset bootstrap

**Files:**
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/assets.py`
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/models/tfv6.py`
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/models/interfuser.py`
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/models/diffusiondrive_v2.py`
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/models/sparsedrive_v2.py`
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/models/select.py`
- Create: `src/Validator_pkg/validator_inference/scripts/fetch_model_assets.py`
- Create: `src/Validator_pkg/validator_inference/test/test_assets_and_model_selection.py`

**Interfaces:**
- Consumes: selected profile `source`/`runtime_assets`, `models/sources/{lead,interfuser,diffusiondrive_v2,sparsedrive_v2}` at pinned commits, and checkpoint assets at profile paths.
- Produces: `build_selected_model(profile)`, returning a concrete `TFv6Model`, `InterFuserModel`, `DiffusionDriveV2InferenceModel`, or `SparseDriveV2Model`; no generic adapter class exists.

- [ ] **Step 1: Write failing asset/model-selection tests**

```python
from validator_inference.models.select import build_selected_model


def test_factory_returns_only_concrete_model_classes(fake_profile):
    assert type(build_selected_model(fake_profile("tfv6"))).__name__ == "TFv6Model"
    assert type(build_selected_model(fake_profile("interfuser"))).__name__ == "InterFuserModel"
    assert type(build_selected_model(fake_profile("diffusiondrive_v2"))).__name__ == "DiffusionDriveV2InferenceModel"
    assert type(build_selected_model(fake_profile("sparsedrive_v2"))).__name__ == "SparseDriveV2Model"


def test_asset_check_reports_path_and_expected_hash(tmp_path):
    with pytest.raises(AssetError, match="expected sha256"):
        verify_runtime_assets({"runtime_assets": [{"path": str(tmp_path / "missing.pth"), "sha256": "0" * 64}]})
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_assets_and_model_selection.py -q`

Expected: imports fail because assets and models do not exist.

- [ ] **Step 3: Implement asset verification and source checkout bootstrap**

```python
def verify_runtime_assets(profile: dict[str, Any]) -> None:
    for asset in profile["runtime_assets"]:
        path = Path(asset["path"])
        if not path.is_file():
            raise AssetError(f"missing runtime asset: {path}")
        expected = asset.get("sha256")
        if expected and sha256_file(path) != expected:
            raise AssetError(f"asset {path} expected sha256 {expected}")


def build_selected_model(profile: SelectedProfile) -> object:
    return {
        "tfv6": TFv6Model,
        "interfuser": InterFuserModel,
        "diffusiondrive_v2": DiffusionDriveV2InferenceModel,
        "sparsedrive_v2": SparseDriveV2Model,
    }[profile.model_id](profile)
```

`fetch_model_assets.py --model-id <id>` clones each profile repository to `models/sources/<id>` at its exact profile commit, downloads every profile asset into its declared `path`, checks every non-null SHA-256, computes and records declared post-download hashes, and writes `models/<id>/asset-manifest.json` containing role/path/hash/source/commit. For DiffusionDrive it copies the official `kmeans_navsim_traj_20.npy` source file into `models/diffusiondrive_v2/navsim_v2/`; it never downloads PDM metric-cache. For TFv6 it checks out the verified compatible public `cvpr2026` pin and downloads `noradar_resnet34` config plus the two ensemble weights. The script exits nonzero on one missing asset or hash mismatch.

- [ ] **Step 4: Implement each source-native forward boundary**

```python
class DiffusionDriveV2InferenceModel:
    def forward(self, frame: InferenceFrame, rng_seed: int) -> ModelPrediction:
        self._set_seed(rng_seed)
        trajectory = self._source_model.forward_test_rl_without_pdm(
            img=frame.inputs["camera_main"],
            lidar=frame.inputs["lidar_bev"],
            status=frame.inputs["driving_feature"].status_feature,
        )
        return ModelPrediction(trajectory=trajectory[:, -1])
```

`TFv6Model.forward` uses the checked-out LEAD source temporal LiDAR builder with `frame.inputs["lidar_raw_history"]`, the panorama image, speed/command, and three route points in that source's input order. The selected official `noradar_resnet34` config does not build or read radar features. `InterFuserModel.forward` directly calls its checked-out agent model with the four published camera tensors, BEV, command/speed, and current target point. `SparseDriveV2Model.forward` directly constructs the source data dictionary with three image tensors, NAVSIM status, and `lidar2img` calculated from the inline calibration messages; it applies no calibration/image concatenation. `DiffusionDriveV2InferenceModel` is the explicitly named source derivative that removes PDM imports, process pool construction, metric cache, evaluator invocation, and `reward_dict`, retaining the checkpoint's diffusion/coarse/top-32/three-stage fine selector. All imports are made only after `verify_runtime_assets` reports success.

- [ ] **Step 5: Run model selection and asset failure coverage**

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_assets_and_model_selection.py -q`

Expected: factory tests pass and the missing asset error names the exact path and hash.

- [ ] **Step 6: Commit concrete runtime preparation**

```bash
git add src/Validator_pkg/validator_inference/src/validator_inference/assets.py src/Validator_pkg/validator_inference/src/validator_inference/models src/Validator_pkg/validator_inference/scripts/fetch_model_assets.py src/Validator_pkg/validator_inference/test/test_assets_and_model_selection.py
git commit -m "feat: add source-native model runtimes"
```

### Task 6: Add no-queue worker, deterministic seed, and coordinate-correct path builder

**Files:**
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/worker.py`
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/path_builder.py`
- Create: `src/Validator_pkg/validator_inference/test/test_worker_and_path.py`

**Interfaces:**
- Consumes: `InferenceFrame`, one concrete model object's `forward`, `run_id`, profile output transform, and `T_base_from_lidar`.
- Produces: `Completion(result, elapsed_ms, rng_seed)` or `REJECTED_BUSY`; `nav_msgs/Path` in `Lidar3D-4`.

- [ ] **Step 1: Write failing concurrency and transform tests**

```python
def test_worker_rejects_a_second_frame_while_forward_is_running(blocking_model, frame):
    worker = InferenceWorker(blocking_model, "run-a")
    assert worker.try_submit(frame).accepted is True
    assert worker.try_submit(frame).reason == "busy_tick_skipped"


def test_carla_path_flips_y_then_uses_inverse_base_from_lidar():
    path = PathMessageBuilder(t_base_from_lidar=translation(1.0, 0.0, 0.0)).build(
        trajectory=[[3.0, 2.0]], coordinate_convention="CARLA ego-relative", stamp=time_from_sec(1.0)
    )
    assert path.header.frame_id == "Lidar3D-4"
    assert (path.poses[0].pose.position.x, path.poses[0].pose.position.y) == (2.0, -2.0)
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_worker_and_path.py -q`

Expected: imports fail.

- [ ] **Step 3: Implement worker mailbox and seed policy**

```python
def seed_from(run_id: str, model_id: str, reference_stamp_ns: int) -> int:
    raw = f"{model_id}:{reference_stamp_ns}:{run_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def try_submit(self, frame: InferenceFrame) -> SubmitResult:
    if not self._forward_lock.acquire(blocking=False):
        return SubmitResult(False, "busy_tick_skipped")
    threading.Thread(target=self._run_forward, args=(frame,), daemon=True).start()
    return SubmitResult(True, "accepted")
```

`_run_forward` measures `time.perf_counter_ns`, calls only the selected concrete model, and atomically replaces the one completion mailbox entry before releasing the lock. `take_completed` removes at most one result. DDV2 receives `seed_from`; the other models report seed zero. `PathMessageBuilder` flips CARLA y but not NAVSIM y, converts base points with `inverse(T_base_from_lidar)`, and fills `PoseStamped.header` plus path header with reference stamp and `Lidar3D-4`.

- [ ] **Step 4: Run worker/path tests**

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_worker_and_path.py -q`

Expected: tests show no second concurrent forward and both coordinate conventions produce `Lidar3D-4` paths.

- [ ] **Step 5: Commit worker and path builder**

```bash
git add src/Validator_pkg/validator_inference/src/validator_inference/worker.py src/Validator_pkg/validator_inference/src/validator_inference/path_builder.py src/Validator_pkg/validator_inference/test/test_worker_and_path.py
git commit -m "feat: add one-slot inference worker and path export"
```

### Task 7: Compose `InferenceEngineNode` and publish every outcome

**Files:**
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/engine.py`
- Create: `src/Validator_pkg/validator_inference/scripts/inference_engine_node.py`
- Create: `src/Validator_pkg/validator_inference/test/test_engine.py`

**Interfaces:**
- Consumes: `SelectedProfile`, `InputBuffer`, `InferenceWorker`, `PathMessageBuilder`, ROS clock time, and `DrivingFeature.header.stamp` as reference stamp.
- Produces: `/waypoints` and one `/inference_metric` message for every completed, not-ready, or busy timer tick.

- [ ] **Step 1: Write failing engine outcome tests**

```python
def test_not_ready_tick_emits_metric_with_missing_keys(engine):
    engine.on_timer_tick(timer_event_at(10.0))
    metric = engine.metric_publisher.messages[-1]
    assert metric.outcome == "not_ready"
    assert metric.missing_input_keys == ["camera_main"]


def test_busy_tick_emits_metric_without_waiting(engine_with_busy_worker):
    engine_with_busy_worker.on_timer_tick(timer_event_at(10.0))
    assert engine_with_busy_worker.metric_publisher.messages[-1].outcome == "busy_tick_skipped"


def test_completion_publishes_path_time_and_completed_metric(engine_with_completion):
    engine_with_completion.on_timer_tick(timer_event_at(10.0))
    assert engine_with_completion.path_publisher.messages[-1].header.frame_id == "Lidar3D-4"
    assert engine_with_completion.time_publisher.messages[-1].data == 12.5
    assert engine_with_completion.metric_publisher.messages[-1].outcome == "inference_completed"
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_engine.py -q`

Expected: import failure for `validator_inference.engine`.

- [ ] **Step 3: Implement timer order and subscriber ownership**

```python
def on_timer_tick(self, _event: rospy.timer.TimerEvent) -> None:
    self._tick_index += 1
    completion = self._worker.take_completed()
    if completion is not None:
        self._publish_completion(completion)
    reference_stamp = self._buffer.latest_driving_feature_stamp()
    snapshot = self._buffer.snapshot_at(reference_stamp, self._required_keys)
    if isinstance(snapshot, NotReady):
        self._publish_metric("not_ready", reference_stamp, missing=snapshot.missing_input_keys)
        return
    submitted = self._worker.try_submit(snapshot)
    if not submitted.accepted:
        self._publish_metric("busy_tick_skipped", reference_stamp)
```

`inference_engine_node.py` uses `rospy.Timer(rospy.Duration.from_sec(0.1), engine.on_timer_tick)`, makes all `rospy.Subscriber` instances inside `InputBuffer.configure(profile)`, and uses a type map limited to declared profile types: `sensor_msgs/Image`, `sensor_msgs/PointCloud2`, `geometry_msgs/PointStamped`, `validator_msgs/DrivingFeature`, and `tf2_msgs/TFMessage`. It starts with `use_sim_time` true and exits immediately if it is false. `_publish_completion` publishes a path and an `InferenceMetric` with matching run/tick/header metadata and the measured `elapsed_ms`; `not_ready` and `busy_tick_skipped` use `elapsed_ms = -1.0`.

- [ ] **Step 4: Run engine tests**

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_engine.py -q`

Expected: all three outcomes publish exactly one metric; only completion has a measured nonnegative `elapsed_ms`.

- [ ] **Step 5: Commit engine composition**

```bash
git add src/Validator_pkg/validator_inference/src/validator_inference/engine.py src/Validator_pkg/validator_inference/scripts/inference_engine_node.py src/Validator_pkg/validator_inference/test/test_engine.py
git commit -m "feat: publish open-loop inference outcomes"
```

### Task 8: Implement startup CSV recording

**Files:**
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/csv_recorder.py`
- Create: `src/Validator_pkg/validator_inference/scripts/recorder_node.py`
- Create: `src/Validator_pkg/validator_inference/test/test_csv_recorder.py`

**Interfaces:**
- Consumes: `/inference_metric` only.
- Produces: one startup-created CSV in default `records/openloop` and exactly one row per metric message.

- [ ] **Step 1: Write failing CSV tests**

```python
def test_filename_uses_startup_wall_time_and_model_id(tmp_path):
    writer = CsvRunWriter(tmp_path, "tfv6", wall_clock=fixed_wall_clock("2026-07-21T10:11:12+09:00"))
    assert writer.path.name == "inference_20260721T101112+0900_tfv6.csv"


def test_metric_creates_one_row_without_a_secondary_timing_topic(tmp_path, metric):
    writer = CsvRunWriter(tmp_path, "tfv6", wall_clock=fixed_wall_clock())
    writer.append_metric(metric)
    assert list(csv.DictReader(writer.path.open()))[0]["outcome"] == "busy_tick_skipped"
    assert len(list(csv.DictReader(writer.path.open()))) == 1
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_csv_recorder.py -q`

Expected: import failure for `validator_inference.csv_recorder`.

- [ ] **Step 3: Implement writer and ROS recorder node**

```python
CSV_COLUMNS = [
    "run_id", "run_started_at", "model_id", "profile_sha256", "tick_index", "tick_wall_time",
    "reference_stamp", "outcome", "elapsed_ms", "missing_input_keys", "compatibility_flags",
    "history_count", "history_span_ms", "rng_seed", "input_keys", "input_age_ms", "calibration_profile_sha256",
]


def append_metric(self, metric: InferenceMetric) -> None:
    row = {
        "run_id": metric.run_id, "run_started_at": self.run_started_at,
        "model_id": metric.model_id, "profile_sha256": metric.profile_sha256,
        "tick_index": metric.tick_index, "tick_wall_time": datetime.now().astimezone().isoformat(),
        "reference_stamp": f"{metric.header.stamp.secs}.{metric.header.stamp.nsecs:09d}",
        "outcome": metric.outcome, "elapsed_ms": metric.elapsed_ms,
        "missing_input_keys": json.dumps(metric.missing_input_keys),
        "compatibility_flags": json.dumps(metric.compatibility_flags),
        "history_count": metric.history_count, "history_span_ms": metric.history_span_ms,
        "rng_seed": metric.rng_seed, "input_keys": json.dumps(metric.input_keys),
        "input_age_ms": json.dumps(metric.input_age_ms),
        "calibration_profile_sha256": metric.calibration_profile_sha256,
    }
    self._dict_writer.writerow(row)
    self._file.flush()
```

`recorder_node.py` opens the writer before subscribing, resolves the default directory from package root plus `records/openloop`, subscribes only to `/inference_metric`, and calls `append_metric` exactly once in each metric callback.

- [ ] **Step 4: Run CSV tests**

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_csv_recorder.py -q`

Expected: each metric produces one durable row; no secondary timing subscriber exists.

- [ ] **Step 5: Commit recorder implementation**

```bash
git add src/Validator_pkg/validator_inference/src/validator_inference/csv_recorder.py src/Validator_pkg/validator_inference/scripts/recorder_node.py src/Validator_pkg/validator_inference/test/test_csv_recorder.py
git commit -m "feat: record inference metrics to per-run CSV"
```

### Task 9: Add launch/run documentation and deterministic rosbag smoke coverage

**Files:**
- Create: `src/Validator_pkg/validator_inference/launch/openloop_inference.launch`
- Create: `src/Validator_pkg/validator_inference/test/test_openloop_smoke.py`
- Modify: `README.md`
- Modify: `docs/inference_engine_design.md`
- Modify: `docs/model_io_contracts.md`

**Interfaces:**
- Consumes: built message and inference packages, fetched assets, complete SparseDrive calibration, active PreprocessLayer topics, and the recorded rosbag.
- Produces: one reproducible command for each model and an automated assertion that a run emits CSV outcomes.

- [ ] **Step 1: Write a failing launch-contract test**

```python
from pathlib import Path


def test_launch_uses_sim_time_and_starts_all_three_nodes():
    launch = Path("src/Validator_pkg/validator_inference/launch/openloop_inference.launch").read_text()
    assert '<param name="/use_sim_time" value="true"/>' in launch
    assert 'type="opt_feeder_node.py"' in launch
    assert 'type="inference_engine_node.py"' in launch
    assert 'type="recorder_node.py"' in launch
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_openloop_smoke.py -q`

Expected: `FileNotFoundError` because the launch file does not exist.

- [ ] **Step 3: Add the launch file and exact operator run sequence**

```xml
<launch>
  <arg name="model_id" default="interfuser"/>
  <arg name="record_dir" default="$(find validator_inference)/../../records/openloop"/>
  <param name="/use_sim_time" value="true"/>
  <node pkg="validator_inference" type="opt_feeder_node.py" name="opt_feeder" output="screen">
    <param name="model_id" value="$(arg model_id)"/>
  </node>
  <node pkg="validator_inference" type="inference_engine_node.py" name="inference_engine" output="screen">
    <param name="model_id" value="$(arg model_id)"/>
  </node>
  <node pkg="validator_inference" type="recorder_node.py" name="recorder" output="screen">
    <param name="model_id" value="$(arg model_id)"/>
    <param name="record_dir" value="$(arg record_dir)"/>
  </node>
</launch>
```

Document the operational sequence exactly:

```bash
catkin_make
source devel/setup.bash
rosrun validator_inference fetch_model_assets.py --model-id interfuser
roslaunch validator_inference openloop_inference.launch model_id:=interfuser
rosbag play --clock rosbag/2026-07-16-17-13-14.bag
```

The smoke test launches with a fake concrete model and published preprocessed fixtures, then asserts the created CSV has at least one `not_ready` or `inference_completed` row and uses `Lidar3D-4` for every published path. It also asserts that a sparse start with the shipped null calibration fails before a worker starts, naming the missing calibration field.

- [ ] **Step 4: Run launch, unit, and smoke verification**

Run: `PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_msgs/test src/Validator_pkg/validator_inference/test -q && catkin_make --pkg validator_msgs && catkin_make --pkg validator_inference && git diff --check`

Expected: all tests pass, both packages build, and `git diff --check` prints no whitespace errors.

- [ ] **Step 5: Commit runnable integration**

```bash
git add src/Validator_pkg/validator_inference/launch src/Validator_pkg/validator_inference/test/test_openloop_smoke.py README.md docs/inference_engine_design.md docs/model_io_contracts.md
git commit -m "feat: add reproducible open-loop inference launch"
```

## Self-Review

### Spec coverage

- Local selected profile, no remote fetch: Task 2.
- `InputBuffer` subscriptions, causal inputs, ages, 10-sweep TFv6 history, and no stale skip: Task 3 and Task 7.
- `/Ego_topic` feature publication, fallback command/route, TFv6 empty optional radar fields, and inline Sparse calibration: Task 4.
- Four source-native models, exact assets, TFv6 official no-radar behavior, DDV2 PDM-free selector, and anchor colocation: Task 5.
- One worker/no queue, busy metric, deterministic DDV2 seed, model-coordinate export: Task 6.
- 10 Hz engine and topic semantics: Task 7.
- Startup timestamped per-run CSV and one-row-per-metric rule: Task 8.
- `/use_sim_time`, `rosbag --clock`, launch, and end-to-end smoke validation: Task 9.

### Placeholder scan

The plan contains no deferred implementation markers. Each task names exact files, public interfaces, failing test, implementation behavior, passing command, and commit boundary.

### Type consistency

`SelectedProfile` is created in Task 2 and consumed in Tasks 4, 5, and 7. `InputBuffer.snapshot_at` produces the `InferenceFrame`/`NotReady` union defined in Task 3 and consumed in Tasks 5–7. All nodes communicate only through the message types introduced in Task 1.
