# Inference Model Contract Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finalize the local inference profile and documentation so the four selected checkpoints have source-derived, executable open-loop I/O contracts.

**Architecture:** `profile.json` is the only local runtime configuration source. Binary model assets remain beside their checkpoint in `models/`; static SparseDrive calibration values remain inline in the selected profile. The engine design describes ROS objects, while a new I/O reference captures actual tensor and asset contracts.

**Tech Stack:** ROS1 (`rospy`), JSON, Mermaid, Markdown, PyTorch reference source.

## Global Constraints

- Work only in the inference layer; use no remote model-profile lookup and no generic `ModelAdapter`.
- `InputBuffer` owns subscriptions; the main loop snapshots at 10 Hz and records stale-input ages.
- Every outcome is published as `validator_msgs/InferenceMetric`; its `elapsed_ms` is milliseconds for a completed forward and `-1.0` when no forward ran. `RecorderNode` writes it to a startup-created CSV.
- Use `Lidar3D-4` for ROS path and temporary target-point frames.
- Omit NAVSIM PDM, metric-cache, and `reward_dict` from DiffusionDrive V2 waypoint inference.
- TFv6 uses the official `noradar_resnet34` release.  It requires panorama, ten LiDAR sweeps with paired poses, command/speed, and previous/current/next target points; it has no radar model input.  Its optional `DrivingFeature` radar fields remain empty with `rows=0` and `cols=0`, with no padding operation or `synthetic_radar_empty` flag.  The LEAD SHA `730bc1a2f44d5f28312dd55f0ca958e94a24c038` is a verified-compatible public `cvpr2026` pin, not a claim about private training provenance.
- Keep SparseDrive V2 calibration as nullable inline profile fields. Do not create a separate calibration file.

---

### Task 1: Finalize executable profile contracts

**Files:**

- Modify: `src/Validator_pkg/config/profile.json`
- Test: repository-root Python JSON assertion

**Interfaces:**

- Consumes: official TFv6 no-radar and DiffusionDrive V2 SEL trajectory-return contracts.
- Produces: selected-profile startup data for `OptFeederNode`, `InputBuffer`, and each model object.

- [ ] **Step 1: Write the profile assertions**

```python
assert tfv6["reference_variant"] == "ln2697/tfv6 noradar_resnet34"
assert tfv6["source"]["commit"] == "730bc1a2f44d5f28312dd55f0ca958e94a24c038"
assert tfv6["radar_input"]["enabled"] is False
assert tfv6["radar_input"]["config_field"] == "use_radar_detection=false"
assert "pdm_metric_cache" not in str(diffusiondrive_v2)
assert diffusiondrive_v2["output"]["published_forward_output"] == "trajectory"
assert sparsedrive_v2["static_runtime_inputs"]["camera_calibration"]["source"] == "profile_inline"
```

- [ ] **Step 2: Edit the profile with exact contracts**

Set TFv6 to the official `noradar_resnet34` contract under verified-compatible public LEAD `cvpr2026` pin `730bc1a2f44d5f28312dd55f0ca958e94a24c038`.  It requires panorama, ten LiDAR sweeps with paired poses, command/speed, and previous/current/next target points, but no radar input.  Keep optional `DrivingFeature` radar empty with `rows=0` and `cols=0`; do not synthesize padded radar data and do not record `synthetic_radar_empty`.  Record that the source SHA is compatibility evidence rather than a provable private training commit.

Set DiffusionDrive V2's `trajectory_anchor` to `models/diffusiondrive_v2/navsim_v2/kmeans_navsim_traj_20.npy` with SHA-256 `2d341e525d7882ee5e916c67d37bea571e4c793cf801f0fff3eff59d38b20d98`. Remove PDM cache from assets and inputs; declare the direct source trajectory expression `traj_to_score[:, -1]`.

Replace SparseDrive V2's calibration file path with `static_runtime_inputs.camera_calibration`. Add three records for `Camera-1`, `Camera-2`, and `Camera-3`, with `null` values for intrinsic, distortion model/coefficients, and `camera_to_lidar_4x4`.

- [ ] **Step 3: Run the profile check**

Run:

```bash
python3 -c 'import json; p=json.load(open("src/Validator_pkg/config/profile.json")); q={x["id"]:x for x in p["profiles"]}; t=q["tfv6"]; assert t["reference_variant"]=="ln2697/tfv6 noradar_resnet34"; assert t["source"]["commit"]=="730bc1a2f44d5f28312dd55f0ca958e94a24c038"; assert t["radar_input"]["enabled"] is False; assert "pdm_metric_cache" not in str(q["diffusiondrive_v2"]); assert q["sparsedrive_v2"]["static_runtime_inputs"]["camera_calibration"]["source"]=="profile_inline"; print("PROFILE_OK")'
```

Expected: `PROFILE_OK`.

- [ ] **Step 4: Commit the profile contract**

```bash
git add src/Validator_pkg/config/profile.json
git commit -m "docs: finalize inference model profile contracts"
```

### Task 2: Correct the inference architecture document

**Files:**

- Modify: `docs/inference_engine_design.md`
- Test: Mermaid CLI render

**Interfaces:**

- Consumes: profile keys from Task 1.
- Produces: valid Mermaid diagrams and readiness rules that agree with the runtime profile.

- [ ] **Step 1: Correct readiness and asset statements**

Replace TFv6 real-radar blocking language with the official no-radar `noradar_resnet34` input contract: panorama, ten LiDAR sweeps/poses, command/speed, and previous/current/next target points only.  Optional `DrivingFeature` radar stays empty (`rows=0`, `cols=0`) and no synthetic padding or compatibility flag is allowed.  Preserve the source-pin nuance: the public LEAD `cvpr2026` SHA is verified-compatible, not evidence of a private training commit.  Replace DiffusionDrive V2 PDM blocking language with the direct source trajectory-return path. Describe the official anchor as a required binary asset stored next to the checkpoint, not as a user-supplied artifact.

- [ ] **Step 2: Correct OptFeeder and calibration ownership**

Make `OptFeederNode` publish all model-required non-image/non-LiDAR fields: ego features, fallback route, and SparseDrive inline calibration.  For TFv6 the optional radar portion of `DrivingFeature` is empty (`rows=0`, `cols=0`); it is not a synthetic-radar producer. Remove every reference to a separate SparseDrive calibration JSON asset.

- [ ] **Step 3: Preserve renderable diagrams**

Keep actual engine objects as Mermaid participants, external systems as layers, and no subscribe callback self-loops. Use literal `\\n` only inside Mermaid node labels where a line break is needed.

- [ ] **Step 4: Render the design document**

Run:

```bash
PUPPETEER_EXECUTABLE_PATH=/private/tmp/puppeteer-browser-cache/chrome-headless-shell/mac_arm-150.0.7871.24/chrome-headless-shell-mac-arm64/chrome-headless-shell PUPPETEER_SKIP_DOWNLOAD=true npx --yes --cache /private/tmp/mermaid-npm-cache --package @mermaid-js/mermaid-cli mmdc -i docs/inference_engine_design.md -o /private/tmp/inference_design_render.md -a /private/tmp/inference_design_artifacts -q
```

Expected: command exits `0`.

- [ ] **Step 5: Commit the design correction**

```bash
git add docs/inference_engine_design.md
git commit -m "docs: align inference design with model contracts"
```

### Task 3: Add source-derived model I/O documentation

**Files:**

- Create: `docs/model_io_contracts.md`
- Modify: `README.md`
- Modify: `docs/inference_engine_design.md`
- Test: `rg` document-link check

**Interfaces:**

- Consumes: official model source/config contracts and the approved compatibility policy.
- Produces: standalone reference for the implementation team.

- [ ] **Step 1: Write `docs/model_io_contracts.md`**

For every model, state the source/commit, binary assets, exact input tensors, preprocessing, `/driving_feature` contribution, output tensor, coordinate conversion, current bag availability, and any compatibility flag.

Describe DiffusionDrive V2's exact inference flow: the official `(20, 8, 2)` anchor creates candidate trajectories; coarse scoring retains 32; the three-stage fine scorer chooses one final candidate; `traj_to_score[:, -1]` is the exported `(8, 3)` trajectory. Define PDM and `reward_dict` as excluded evaluation functionality.

- [ ] **Step 2: Link the reference**

Add `docs/model_io_contracts.md` beside the inference design link in `README.md`, and add a reciprocal link near the design document introduction.

- [ ] **Step 3: Verify discoverability**

Run:

```bash
rg -n 'model_io_contracts|Inference Engine 설계' README.md docs/inference_engine_design.md docs/model_io_contracts.md
```

Expected: all three files report the document link.

- [ ] **Step 4: Commit the reference**

```bash
git add README.md docs/inference_engine_design.md docs/model_io_contracts.md
git commit -m "docs: add source-derived model I/O contracts"
```

### Task 4: Verify the final documentation and configuration

**Files:**

- Modify: no source files
- Test: JSON parser, Mermaid render, whitespace check, scoped diff

**Interfaces:**

- Consumes: Tasks 1–3.
- Produces: validated artifacts ready for implementation.

- [ ] **Step 1: Validate profile and Mermaid**

Run the Task 1 JSON parser and render both `docs/inference_engine_design.md` and `README.md` with Mermaid CLI. Each command must exit `0`.

- [ ] **Step 2: Check patch hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; unrelated user work such as `src/@morai_msgs` remains untouched.

- [ ] **Step 3: Inspect the scoped final diff**

Run:

```bash
git diff -- README.md docs/inference_engine_design.md docs/model_io_contracts.md src/Validator_pkg/config/profile.json
```

Expected: no PDM dependency, official anchor colocated with its model, inline nullable SparseDrive calibration, and the official no-radar TFv6 behavior (empty optional radar fields, no padding or compatibility flag).

- [ ] **Step 4: Commit any uncommitted planned files**

```bash
git add README.md docs/inference_engine_design.md docs/model_io_contracts.md src/Validator_pkg/config/profile.json
git commit -m "docs: finalize open-loop inference contracts"
```
