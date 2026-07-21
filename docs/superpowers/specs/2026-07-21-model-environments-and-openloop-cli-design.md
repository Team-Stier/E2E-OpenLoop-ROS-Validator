# Model-specific environments and simplified open-loop CLI

## Context

The current integrated command starts every Python node through Catkin-generated
relays whose shebang is `/usr/bin/python3`. On ROS Noetic in this workspace that
is Python 3.8, so activating another virtual environment does not change the
interpreter used by `inference_engine_node.py`. This caused InterFuser to fail
with `No module named 'torch'` even when the required packages existed in a
different environment.

The four pinned model sources do not share one supported Python stack:

| Model | Python | Primary runtime | CUDA policy | Validation in this change |
| --- | --- | --- | --- | --- |
| `interfuser` | CPython 3.8.20 | PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121 | Required for this acceptance test; implementation can otherwise report CPU capability. | Environment, ROS smoke, model load, and bag inference. |
| `tfv6` | CPython 3.10.20 | PyTorch 2.5.0+cu124, torchvision 0.20.0+cu124 | Required by the model runtime. | Definition, lock, command, and prerequisite preflight. |
| `diffusiondrive_v2` | CPython 3.9.25 | PyTorch 2.0.1+cu118, torchvision 0.15.2+cu118 | Recommended; absence warns, not blocks, before model-specific load policy is evaluated. | Definition, lock, command, and prerequisite preflight. |
| `sparsedrive_v2` | CPython 3.9.25 | PyTorch 2.0.1+cu118, torchvision 0.15.2+cu118 | CUDA extension required; Toolkit 11.8 and a compatible host compiler are blockers. | Definition, lock, command, and expected `nvcc`/calibration blockers. |

A shared system interpreter or a cross-environment `PYTHONPATH` therefore cannot
be the supported runtime contract.

## Goals

- Give each supported model an isolated, fully pinned environment definition for
  the target Linux x86_64 host.
- Keep preprocessing, opt feeder, recorder, visualization, and RViz on the ROS
  Noetic Python 3.8 runtime.
- Run only the inference engine with the selected model interpreter.
- Reduce normal operation to one repository-local command plus a separate bag
  playback command.
- Make setup explicit, idempotent, and separate from normal execution.
- Fail before `roslaunch` when an environment, asset, CUDA prerequisite, or
  calibration is missing or stale.
- Complete an actual CUDA and rosbag inference validation for InterFuser.
- Define the setup/run path and preflight all four models without claiming that
  unavailable TFv6, DiffusionDrive, or SparseDrive runtimes were installed or
  GPU-forward validated.

## Non-goals

- Replacing ROS1 or the existing topic and message contracts.
- Moving all ROS nodes into each model environment.
- Adding a model-process IPC protocol or shared-memory transport.
- Automatically installing host GPU drivers, CUDA Toolkit, ROS Noetic, or `uv`.
- Silently downloading dependencies during a normal run.
- Treating a preflight-only model as end-to-end validated.
- Labeling a third-party compatibility mirror as an official checkpoint source.

## User interface

The repository root exposes one executable named `openloop`.

```bash
./openloop setup interfuser
./openloop interfuser --rviz
./openloop bag
```

`./openloop run interfuser --rviz` is an explicit equivalent of the shorthand
second command. `./openloop bag --path <bag>` overrides the bag declared by the
profile. `./openloop verify MODEL` performs a full asset and environment
integrity check. `./openloop setup MODEL --recreate` always builds a new version
without modifying the active environment until validation succeeds. Advanced
users may still invoke the underlying `roslaunch` command once the model
environment exists.

The commands have distinct responsibilities:

- `setup MODEL` builds the Catkin workspace, obtains or verifies declared model
  assets, creates the selected environment, installs its locked dependencies,
  builds model-specific extensions when applicable, creates the ROS overlay,
  and runs preflight checks.
- `run MODEL` and the model shorthand perform read-only preflight checks and
  start the integrated launch. They never install, update, or download files.
- `bag` starts `rosbag play --clock` for the configured or supplied bag path in
  a separate terminal.
- `verify MODEL` recomputes full hashes. Normal runs use the verified state and
  file fingerprints so multi-hundred-megabyte checkpoints do not delay launch.

### Clean-checkout bootstrap

The root `openloop` executable does not depend on a prior Catkin build. It
locates the repository, sources `/opt/ros/noetic/setup.bash`, conditionally
sources `devel/setup.bash` when present, prepends
`src/Validator_pkg/validator_inference/src` to the controller's `PYTHONPATH`,
and executes the source-tree `scripts/openloop_cli.py` with `/usr/bin/python3`.
The system-Python controller may retain the ROS setup paths needed to invoke ROS
tools; strict path sanitization applies only at the cross-version inference
boundary.

The CLI and its environment-management imports must remain ROS-message-free at
module import time. `setup` may therefore invoke `catkin_make` in a clean
checkout and source the newly created devel space for later phases.

## Runtime architecture

The integrated launch retains its existing three layers and topic flow. The
only interpreter boundary is the inference engine node.

```text
roslaunch (ROS Python 3.8)
  |-- preprocessing nodes       (ROS Python 3.8)
  |-- opt_feeder                (ROS Python 3.8)
  |-- recorder                  (ROS Python 3.8)
  |-- visualization / RViz      (ROS Python 3.8 / native)
  `-- model_env_exec            (launch-prefix)
        `-- .venvs/MODEL/current/bin/python inference_engine_node.py
```

`openloop_inference.launch` adds a model environment root argument and applies a
`launch-prefix` only to `inference_engine_node.py`. It passes the selected model
ID and environment root to a small executable wrapper through node-local
environment variables. Node name, node type, private parameters, remappings,
and the three-node inference-layer contract remain unchanged.

The CLI always passes the absolute repository `.venvs` path; launch files do not
infer it by walking upward from `$(find ...)`. For direct advanced `roslaunch`,
the source-workspace default is available and install-space callers must provide
`model_env_root:=<absolute-path>`.

The wrapper validates the model ID and interpreter and derives the active Catkin
prefix from the resolved node executable passed as its first argument. It then
selects `<prefix>/lib/python3/dist-packages`, supporting both devel and arbitrary
install prefixes, and verifies that the project packages exist there. It rebuilds
`PYTHONPATH` from an allowlist containing only the model's ROS overlay and that
active Catkin path; it does not filter and retain arbitrary inherited entries.
It unsets `PYTHONHOME`, `PYTHONUSERBASE`, and foreign environment activation
variables, then sets `VIRTUAL_ENV`, `PATH`, `PYTHONNOUSERSITE`, and the
repository-local Torch cache. Finally it uses `exec` to replace itself with the
selected interpreter.
This preserves ROS remapping arguments, PID semantics, Ctrl-C, exit status, and
the existing `required="true"` shutdown behavior.

## Cross-version ROS boundary

ROS Noetic targets Python 3.8, while three selected model sources require Python
3.9 or 3.10. The inference engine only needs the pure-Python ROS client and
generated message layer. Setup creates a curated overlay containing this audited
closure from the active Noetic installation:

```text
rospy, rosgraph, roslib, genpy, genmsg, catkin,
roscpp, rosgraph_msgs, std_msgs, sensor_msgs,
geometry_msgs, nav_msgs, tf2_msgs, actionlib_msgs
```

Setup copies from `/opt/ros/noetic/lib/python3/dist-packages` into the selected
environment's `.ros-overlay/`; it does not symlink a whole source package. Only
package directories, `__init__.py`, and recursively discovered `.py` modules in
the allowlist may be copied. `.so`, `.pyc`, native libraries, and arbitrary
distribution metadata are rejected. Every copied file is recorded with its
source path and SHA256 in `.ros-overlay-manifest.json`, whose digest is included
in the environment state.

The active Catkin devel- or install-space Python path is added separately for
`validator_inference` and project message packages. The wrapper removes
the broad Noetic and system dist-packages paths before adding the curated
overlay. Target-Python builds of `PyYAML`, `rospkg`, `catkin_pkg`, `netifaces`,
and `python-dateutil` live in each model environment. Setup performs a static
import audit over the copied modules and a runtime import smoke; an unresolved
module expands the explicit allowlist in review rather than silently exposing
the source root.

The current `tf.transformations.quaternion_slerp` import transitively loads
`tf2_py/_tf2.so`, which is linked to Python 3.8. The inference package replaces
that one use with a local NumPy quaternion slerp implementation and tests it
against the existing behavior. No Python 3.8 ROS extension module may be loaded
inside a Python 3.9 or 3.10 inference process. In particular, the design excludes
`tf2_py`, `cv_bridge`, and `roslz4` from the overlay.

Runtime preflight checks every imported Python extension module. Its origin must
be inside the selected venv or that interpreter's own standard-library
`lib-dynload`; no CPython extension may originate from `/opt/ros`, system
dist-packages, another venv, or another project. ROS package origins must resolve
to `.ros-overlay`. Generated project messages and the top-level Catkin package
must resolve to the active Catkin prefix; implementation modules may resolve to
this repository's source tree only through a verified devel-space relay, or
directly to the same active install prefix in install space.

The Python 3.11 prototype on this host already imports `rospy`, the required
standard messages, and generated `validator_msgs` when target-version PyYAML,
rospkg, netifaces, and python-dateutil are present. The implementation must
repeat the smoke against the final copied overlay after removing the TF import.

This is a deliberately narrow compatibility layer, not a claim that arbitrary
ROS Noetic Python packages support newer Python versions. If the full ROS smoke
test fails for a selected interpreter, implementation stops at that boundary;
the fallback is a separately designed ROS-driver/model-worker IPC architecture,
not expansion of the overlay with binary packages.

## Environment definition

A tracked `model_environments.json` is the source of truth for:

- allowed model IDs;
- exact CPython patch version and target platform;
- required `uv` version (`0.11.14`) and lock format;
- input and compiled lock file paths;
- ordered Python package indexes and exact PyTorch CUDA wheel tag;
- CUDA policy (`required`, `recommended`, or `extension_required`);
- required imports and model-specific setup actions;
- runtime asset and calibration preflight policy.

The target is Ubuntu 20.04-compatible Linux x86_64. Lock files are produced by
the pinned `uv` with hashes enabled and synchronized with hash verification and
no dependency re-resolution. The manifest records the compile and sync
arguments so lock regeneration is reviewable. PyTorch indexes are fixed to
`https://download.pytorch.org/whl/cu121`, `cu124`, or `cu118` as shown in the
model table; PyPI is the secondary runtime dependency index. SparseDrive also
declares CUDA Toolkit 11.8, `CUDA_HOME`, the accepted `nvcc` 11.8 release, host
compiler compatibility, its single `navsim/agents/sparsedrive/ops` build
directory and command, and import checks for both built extension modules.

CUDA policy has executable semantics:

| Policy | Setup without CUDA | Run without CUDA |
| --- | --- | --- |
| `required` | Block before environment or asset downloads. | Block before launch. |
| `recommended` | Complete with a recorded warning when the model supports CPU. | Warn and continue; acceptance may still require GPU. |
| `extension_required` | Block before downloads unless the declared Toolkit, compiler, and build inputs are present. | Block unless the compiled modules and a CUDA device pass preflight. |

TFv6 uses `required`, DiffusionDrive V2 uses `recommended`, SparseDrive V2 uses
`extension_required`, and InterFuser uses `recommended` while this change's
end-to-end acceptance explicitly requires CUDA.

Each model has a human-maintained input requirements file and a generated,
fully pinned lock file. The pinned source checkout is not installed into the
environment: the existing model wrapper verifies its commit and prepends its
local path at load time. The lock therefore contains only runtime dependencies
and remains independent of the absolute workspace location.

Environments are versioned and excluded from Git:

```text
.venvs/MODEL/
  current -> builds/BUILD_ID
  builds/BUILD_ID/
    bin/python
    .ros-overlay/
    .ros-overlay-manifest.json
    .validator-environment.json
```

`BUILD_ID` is unique and the definition hash is stored in the state file. Each
successful build writes `.validator-environment.json` with at least:

- model ID, full Python version, platform, and selected interpreter path;
- `uv` version and installed-package freeze digest;
- environment schema version;
- manifest and lock SHA256 values;
- ROS overlay manifest SHA256;
- pinned source commit;
- asset SHA256 values plus size and nanosecond-mtime fingerprints;
- setup completion time;
- successful preflight identifiers.

Setup is idempotent. A matching active state skips dependency installation. A
changed definition or `--recreate` creates a new build directory and validates
it without touching `current`. Only after success does setup atomically replace
the `current` symlink. A failed build is never selected; the former environment
continues to run. Inactive builds are retained for diagnosis until an explicit
`./openloop prune MODEL` command removes them. A normal run never mutates any
environment.

## Setup flow

`openloop setup MODEL` performs ordered phases and reports the phase that fails:

1. Validate the model ID, local profile, host OS, ROS Noetic setup, `uv`, and
   model-specific host prerequisites.
2. Fail early for known host blockers. SparseDrive, for example, checks `nvcc`
   and complete camera calibration before downloading or compiling packages.
3. Build the Catkin workspace and verify generated project messages.
4. Fetch or verify the profile-pinned source and runtime assets.
5. Create a unique candidate under `.venvs/MODEL/builds/` with the manifest's
   exact `uv`-managed Python.
6. Synchronize the selected runtime dependency lock.
7. Build any declared native extension and create the pure ROS overlay.
8. Start a temporary `roscore` on an unused localhost port, run the
   target-interpreter ROS smoke with a bounded timeout, and always terminate the
   temporary master and children.
9. Run dependency, CUDA, asset, and model-specific preflights.
10. Write the environment state and atomically select the candidate through the
    `current` symlink only after every required check succeeds.

A failed setup never changes `current`. A subsequent setup can resume safe
download caches but creates a fresh candidate and reruns all validation.

## Assets

Runtime assets remain governed by `profile.json` and the existing asset manifest
and checksum machinery. Environment state refers to those manifests rather than
duplicating checkpoint ownership.

InterFuser has an undeclared ResNet50D initialization download in the pinned
source. The design promotes `resnet50d_ra2-464e36ba.pth` to a declared runtime
asset with SHA256
`464e36baa940a80971e717454e61e8f6fc1a1b284b7a3dbb6f6fd4c3da78533d`
at
`models/shared/torch-cache/hub/checkpoints/resnet50d_ra2-464e36ba.pth`.
The wrapper sets `TORCH_HOME=<project-root>/models/shared/torch-cache`, making
model loading deterministic.

The existing official InterFuser checkpoint URL is currently unavailable. The
target workspace's checkpoint came from the compatibility mirror declared by
`MingfeiCheng/Drivora` at commit
`25d56d2300f422d502c6549695b9e49efd5e5e00`, whose install script names Google
Drive object `1GKiASmGPbD4FwHkUoVfGRk_lMLrGb2f6`. Its observed SHA256 is
`39548dcfa56cdc06b8ccc52d7f6187a7d70ef36b05b7027db4d64a57ac3e30b2`, and it
has passed a strict state-dict load against the profile-pinned InterFuser source.
The profile records this as a third-party compatibility artifact, not as an
officially published checkpoint. This change may use the existing matching
local file for acceptance. If it is absent, setup reports the expected path and
provenance rather than adding an implicit downloader in this scope.

## Run flow and error handling

Before starting `roslaunch`, `run` validates:

- Catkin setup and generated messages;
- model ID and profile selection;
- environment interpreter and exact Python version;
- completion stamp, manifest hash, and lock hash;
- source commit, asset state, and size/mtime fingerprints recorded by the most
  recent deep verification;
- model-specific GPU, CUDA Toolkit, extension, and calibration requirements;
- ROS overlay contents and absence of disallowed ROS binary modules.

Failures are concise and actionable. A missing or stale environment exits before
starting ROS and prints one recovery command:

```text
interfuser environment is missing or stale.
Run: ./openloop setup interfuser
```

Setup failures name the failed phase and underlying prerequisite. Unsupported
model IDs are rejected by the CLI before filesystem changes. Bag playback checks
that the file exists before invoking `rosbag`.

`verify MODEL` and `setup MODEL` compute full SHA256 values for sources declared
by file hash and all runtime assets. `run` avoids hashing large checkpoints on
every invocation: if a recorded size or nanosecond mtime changed, it refuses to
launch and requests `./openloop verify MODEL` or setup. The preflight target is
under two seconds before `roslaunch` on an unchanged workspace.

Once launch begins, existing ROS behavior remains authoritative. A model-load
failure terminates the required inference node and therefore the integrated
launch. The prefix wrapper adds no retry loop and does not hide the original
traceback or exit code.

## Components and ownership

The tracked implementation layout is:

```text
openloop
src/Validator_pkg/validator_inference/
  config/model_environments.json
  requirements/
    interfuser.in
    interfuser.lock
    tfv6.in
    tfv6.lock
    diffusiondrive_v2.in
    diffusiondrive_v2.lock
    sparsedrive_v2.in
    sparsedrive_v2.lock
  scripts/
    openloop_cli.py
    model_env_exec
  src/validator_inference/
    model_environment.py
    environment_setup.py
  test/
    test_model_environment.py
    test_openloop_cli.py
```

The exact test split may reuse an existing contract-test file when that keeps a
single contract in one place, but production ownership remains as listed.

- Root `openloop`: stable shell entry point that locates the repository and uses
  the ROS system interpreter for orchestration.
- Open-loop CLI: parses `setup`, `run`, shorthand model, `verify`, `prune`, and
  `bag`; constructs subprocess arguments without shell interpolation.
- Model environment module: loads and validates the environment manifest,
  computes state hashes, implements preflight decisions, and exposes testable
  command construction.
- Environment setup module: owns `uv` synchronization, overlay construction,
  native extension actions, and atomic completion state.
- `model_env_exec`: minimal launch-prefix executable that validates and `exec`s
  the selected interpreter.
- Launch files: propagate the model environment root and apply the prefix to the
  inference engine only.
- Requirements and locks: model-specific dependency inputs and resolved pins.
- Existing asset/profile modules: remain the authority for source and binary
  asset acquisition and integrity.

The root script always runs the source-tree CLI for clean-checkout bootstrap;
`openloop_cli.py` is additionally installed with `catkin_install_python` for ROS
package tooling. `model_env_exec` is installed without Catkin shebang rewriting
at
`${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts/model_env_exec`; this preserves the
same `$(find validator_inference)/scripts/model_env_exec` launch-prefix path in
source and install spaces. The manifest, requirements, and locks are installed
as package data. The root entry point remains repository-local because it also
owns workspace build and setup discovery.

## Testing

The default unit and contract suite is network-free, GPU-free, and does not
create full model environments. Tests inject subprocess runners, temporary
directories, manifests, and state. Interpreter, install-space, signal, and
model-forward tests are explicitly marked integration tests and are mandatory
release checks for this change rather than implicit work in every unit run.

Unit and contract tests cover:

- manifest schema, all four model definitions, and exact Python selection;
- hash and state-file matching, missing and stale environments, and atomic state;
- versioned candidate creation, atomic `current` switching, failed-build
  preservation, `--recreate`, and explicit pruning;
- command construction without shell injection;
- unknown model and invalid option handling;
- clean-checkout root bootstrap before `devel/setup.bash` exists;
- overlay allowlist construction, per-file manifest hashing, and rejection of
  compiled or undeclared ROS modules;
- prefix argument preservation, exit behavior, and environment sanitization;
- NumPy slerp endpoints, normalization, shortest-path behavior, and numerical
  agreement with the former ROS implementation;
- launch XML proving that only the inference engine receives the prefix;
- integrated launch propagation of model ID and environment root;
- CMake/install layout and executable permissions;
- README and design documentation containing the canonical commands;
- all existing inference, preprocessing, recorder, and layer-integration tests.

ROS compatibility integration tests run independently of model checkpoints and
GPU availability. They create minimal pinned environments for CPython 3.8.20,
3.9.25, and 3.10.20, build the same curated overlay, and start a temporary
`roscore` on an unused port with bounded startup and cleanup timeouts. A system
Python 3.8 peer and each target interpreter verify both directions for the
message types used by the engine:

```text
sensor_msgs/Image, sensor_msgs/PointCloud2,
validator_msgs/DrivingFeature, tf2_msgs/TFMessage,
nav_msgs/Path, validator_msgs/InferenceMetric, rosgraph_msgs/Clock
```

These tests also verify:

- `rospy.init_node`, `/use_sim_time`, and clock advancement;
- imports of the inference engine and model selector;
- ROS import origins are the generated overlay;
- `sys.executable` is the selected venv interpreter;
- `PYTHONHOME` is unset and `PYTHONPATH` contains exactly the generated overlay
  and active Catkin package path;
- every loaded Python extension originates in the selected venv or its own
  standard-library `lib-dynload`, never another project or Python 3.8 ROS path.

An install-space integration test builds into a temporary install prefix and
launches the prefixed stub engine there, proving that the wrapper derives the
install Python path rather than assuming `devel`. A signal integration test uses
real `roslaunch`, sends SIGINT, and verifies remapping arguments, child exit
status, `required=true` shutdown, and absence of surviving wrapper or child
processes.

The InterFuser end-to-end acceptance test uses only
`.venvs/interfuser/current/bin/python`:

1. Complete `./openloop setup interfuser`.
2. Start `./openloop interfuser` without RViz for automation.
3. Replay at least 20 seconds of the configured bag with `--clock`.
4. Verify that the inference engine remains alive and `/waypoints` is published.
5. Verify a new CSV contains at least one `inference_completed` row.
6. Verify the loaded model reports the CUDA device, the exact selected
   interpreter, the generated Python path allowlist, and no external project
   venv.

TFv6, DiffusionDrive V2, and SparseDrive V2 acceptance is limited to manifest,
environment command, and prerequisite preflight coverage. Tests must assert the
specific expected blocker when assets, `nvcc`, or calibration are unavailable.

## Success criteria

- After one successful setup, the normal InterFuser workflow is exactly
  `./openloop interfuser --rviz` plus `./openloop bag` in another terminal.
- No manual environment activation or `PYTHONPATH` export is required.
- InterFuser completes CUDA inference from the configured bag and records it.
- All four model IDs have an exact manifest, hash-locked dependency definition,
  deterministic command selection, and tested prerequisite diagnostics.
- CPython 3.9 and 3.10 pass the model-independent ROS compatibility smoke even
  when their model assets or native toolchains are unavailable.
- Only InterFuser is claimed as installed and end-to-end model validated in this
  change; the other three retain their explicit readiness blockers.
- Missing prerequisites fail before launch with an actionable message.
- A clean checkout can enter `./openloop setup interfuser` without a pre-existing
  devel or install space.
- Existing ROS topics, timing policy, waypoint frame, CSV schema, and launch node
  composition remain unchanged.
