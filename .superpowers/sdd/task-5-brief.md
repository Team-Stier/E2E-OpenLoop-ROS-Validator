# Task 5 — concrete source-native checkpoint runtime and asset bootstrap

## Objective

Implement the local asset bootstrap plus the four **concrete**, source-native model boundaries.  This task must make the inference layer able to locate pinned upstream code and assets deterministically.  It must not introduce a generic `ModelAdapter`, fake neural network, placeholder trajectory, remote profile service, PDM evaluation stack, or a second preprocessing layer.

The selected local profile is authoritative: `src/Validator_pkg/config/profile.json`.  Preserve it unless concrete inspection of the exact pinned source proves a documented contract impossible; in that case stop and report instead of silently altering the model contract.

## Required files

- Create `src/Validator_pkg/validator_inference/src/validator_inference/assets.py`
- Create `src/Validator_pkg/validator_inference/src/validator_inference/models/__init__.py`
- Create `src/Validator_pkg/validator_inference/src/validator_inference/models/tfv6.py`
- Create `src/Validator_pkg/validator_inference/src/validator_inference/models/interfuser.py`
- Create `src/Validator_pkg/validator_inference/src/validator_inference/models/diffusiondrive_v2.py`
- Create `src/Validator_pkg/validator_inference/src/validator_inference/models/sparsedrive_v2.py`
- Create `src/Validator_pkg/validator_inference/src/validator_inference/models/select.py`
- Create `src/Validator_pkg/validator_inference/scripts/fetch_model_assets.py`
- Create `src/Validator_pkg/validator_inference/test/test_assets_and_model_selection.py`

If concise shared model result/error dataclasses are required, put them in a plainly named module (for example `models/contracts.py`); they must not select, normalize, or adapt model inputs generically.

## First inspect the actual pinned sources

Before coding model calls, inspect each exact commit declared in the profile.  You may use a temporary checkout for investigation.  The real bootstrap persists them below ignored `models/sources/` (the checked-in source is **not** vendored).  For TFv6, `730bc1a2f44d5f28312dd55f0ca958e94a24c038` is the verified-compatible public LEAD `cvpr2026` pin for the official `noradar_resnet34` release; it is not claimed to be a provable private training-provenance commit:

- TFv6: profile source repo at pinned commit; directory name may be `lead`.
- InterFuser: `interfuser`.
- DiffusionDrive V2: `diffusiondrive_v2`.
- SparseDrive V2: `sparsedrive_v2`.

For every model wrapper, name the concrete checked-out source file/symbol it imports or invokes, and perform that import only after assets/source checks pass.  Do not invent an API based solely on a paper or make a class with a nonfunctional `forward` that only raises a generic placeholder error.

If an official source has no callable inference boundary independent of CARLA/NAVSIM evaluation, make the minimal explicit source derivative in `models/sources/<model>/` during bootstrap, record its upstream commit and local patch hash in the asset manifest, and have the wrapper import that derivative.  The derivative must retain checkpoint-loaded neural code and cannot synthesize a trajectory.  It may not enter git.  PDM, metric-cache, PDM process pools, evaluators, and `reward_dict` must be absent from the DiffusionDrive V2 forward path.

## `assets.py` contract

Implement pure, testable functions.  Resolve declared relative asset paths from the repository/project root rather than depending on the process CWD.  Failures name the exact missing path and expected hash.

- `sha256_file(path) -> str` reads in chunks.
- `verify_runtime_assets(profile, project_root=...)` verifies every declared runtime asset, including each non-null SHA-256, and returns stable metadata suitable for a manifest.  It must never download.
- Explicitly distinguish a missing asset from a hash mismatch.  An asset with `sha256: null` is allowed but its observed hash is still recorded.
- Validate a source checkout exists and is exactly the profile commit before a concrete source import.
- Write an `asset-manifest.json` inside the selected model asset directory only after all requested assets have been fetched/copied/verified.  It records model ID, source repo URL, exact commit, checkpoint source/revision when applicable, every role/path/declared hash/observed hash, and any explicit local derivative patch hash.  Use canonical deterministic JSON.

## `fetch_model_assets.py` CLI

`rosrun validator_inference fetch_model_assets.py --model-id <one of the four> [--profile-file PATH] [--project-root PATH]`

- Load selected profile locally with `load_selected_profile`; never request it remotely.
- Clone/fetch its source repository to the declared `models/sources/<source-name>` and checkout the exact detached commit.  Reject a pre-existing checkout at another revision rather than quietly using it; a documented `--refresh-source` may explicitly repair it.
- For assets with `source_url`, download to their profile path atomically (temporary sibling then rename).
- For Hugging Face `checkpoint_repository` plus `source_path`, construct the revision-pinned public resolve URL; do not need a user token for listed public assets.
- For `source_repository_path`, copy that exact file from the pinned source checkout (e.g. DDV2 anchor).
- Recheck all hashes via `verify_runtime_assets`, then write the manifest.  A single missing source URL/path, failed download, wrong SHA, or source revision mismatch must make the command exit nonzero and must not leave a manifest implying success.
- Downloaded `models/` remains ignored; no binary, source checkout, or cache is committed.

## Model factory and concrete boundaries

`build_selected_model(selected_profile, project_root=...)` returns exactly one concrete type:

- `TFv6Model`
- `InterFuserModel`
- `DiffusionDriveV2InferenceModel`
- `SparseDriveV2Model`

No generic adapter base class.  Constructors may store the selected profile, but `load()`/first forward must verify assets and source revision before importing heavyweight dependencies.  Do not import PyTorch, CARLA, NAVSIM, or upstream model modules at package import time; the ROS unit suite must remain runnable without the downloaded models.

Each wrapper has a source-specific, named load and forward path, accepting the established `InferenceFrame` and returning the common model prediction contract used by the worker later.  It must validate that exactly the selected profile's inputs are present and name any missing input.

1. **TFv6**: use the pinned LEAD `noradar_resnet34` source temporal LiDAR builder with `lidar_raw_history`, panorama, command/speed, and previous/current/next target in the source-required order.  This official no-radar checkpoint has no radar model input: `DrivingFeature` radar fields must remain empty with `rows=0` and `cols=0`; do not perform source padding and do not emit a `synthetic_radar_empty` compatibility flag.  It produces its checkpoint-native eight-point CARLA-relative prediction; do not recompute preprocessor image features here.
2. **InterFuser**: invoke the pinned source model/agent's actual neural forward with its four distinct images, LiDAR histogram, seven-value command/speed and current target.  It produces its checkpoint-native ten-point CARLA-relative prediction.  No fourth physical camera is fabricated.
3. **DiffusionDrive V2**: use only the source neural path with camera, latest LiDAR BEV, NAVSIM status, and the official local anchor.  Set the worker-supplied deterministic seed before the neural forward.  It exports `traj_to_score[:, -1]` (B×8×3) after the learned fine selector.  It must not import/initialize PDM, evaluator, metric-cache, reward dict, or process pool.
4. **SparseDrive V2**: call the source neural model with left/front/right tensors and NAVSIM status.  Compute source-required projection conditions from `DrivingFeature.camera_calibrations`: `T_camera_from_lidar = inverse(T_lidar_from_camera)` and `lidar2img = A_resize_crop @ K_original @ T_camera_from_lidar[:3,:]`, preserving supplied distortion handling before resize/crop.  Never concatenate calibration into image channels or make raw LiDAR a model input.

Raise a specific runtime error that includes model ID, missing asset/source/dependency/symbol whenever the host lacks optional heavyweight runtime prerequisites.  Never return a zero/default trajectory in that situation.

## Test-first protocol

1. Write tests and run them in the Noetic Docker image before implementation; confirm imports fail first.
2. Cover at least:
   - exact concrete factory types and rejection of unknown type;
   - missing asset and wrong hash errors name path + expected SHA;
   - null declared SHA is accepted but observed hash is returned;
   - project-root-relative resolution is independent of CWD;
   - source checkout revision mismatch is rejected;
   - manifest deterministic contents include source commit and observed hashes;
   - fetch command planning/download dispatch using a local `file://` fixture or mocked downloader/subprocess (never download model binaries in pytest);
   - lazy imports: `build_selected_model` works with no torch/upstream checkout, while `load()` fails with explicit model/source/asset context;
   - concrete forward input validation, including TFv6 history and explicit absence of a radar input (empty `DrivingFeature` radar with `rows=0`, `cols=0`), InterFuser named inputs, DDV2 no PDM import/invocation seam, and Sparse complete calibration/projection conditions.  Use source-specific test doubles at source import boundaries, not an alternate generic adapter.
3. GREEN target in Docker:

```bash
docker compose run --rm ros1-noetic bash -lc 'PYTHONPATH=src/Validator_pkg/validator_inference/src pytest src/Validator_pkg/validator_inference/test/test_assets_and_model_selection.py -q'
```

Then run the full inference package suite and compile the CLI under the same image.

## Scope / safety

- Do not fetch full checkpoints in normal tests.  Do not commit anything under `models/`.
- Do not modify `profile.json`, docs, InputBuffer, OptFeeder, messages, engine/worker/recorder/launch yet, unless a source-derived interface defect makes it impossible to implement; report that rather than broadening scope.
- Preserve user-owned `src/@morai_msgs` and `graphify-out/`.
- Commit with `feat: add source-native model runtimes`.

## Handoff

Report the upstream files/symbols actually inspected and invoked for all four models, RED/GREEN evidence, all source-specific runtime limitations (if any), changed paths, and commit SHA.  Do not self-review.
