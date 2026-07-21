# Model-specific Environments and Open-loop CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build isolated, reproducible Python environments for all four model IDs and reduce normal operation to `./openloop MODEL [--rviz]` plus `./openloop bag`, with InterFuser validated end to end.

**Architecture:** A system-Python controller owns setup and launch while only the inference-engine node crosses into `.venvs/MODEL/current`. Setup creates versioned candidates, copies an audited pure-Python ROS overlay, validates them, and atomically switches `current`; normal runs are read-only and fail before `roslaunch` when state is stale. The existing ROS preprocessing, feeder, recorder, visualizer, topics, and message contracts remain on Noetic Python 3.8.

**Tech Stack:** ROS1 Noetic/catkin, Python 3.8/3.9/3.10, `uv 0.11.14`, PyTorch CUDA wheels, pytest, XML launch files, Bash bootstrap, JSON manifests.

---

## File Map

- Create `openloop`: clean-checkout Bash bootstrap that always invokes the source CLI with `/usr/bin/python3`.
- Create `src/Validator_pkg/validator_inference/config/model_environments.json`: exact environment, CUDA, setup-action, and lock metadata for the four supported models.
- Create `src/Validator_pkg/validator_inference/requirements/*.in`: human-maintained direct runtime inputs.
- Create `src/Validator_pkg/validator_inference/requirements/*.lock`: `uv 0.11.14` generated, hash-checked `x86_64-manylinux_2_31` resolutions.
- Create `src/Validator_pkg/validator_inference/src/validator_inference/quaternion_math.py`: NumPy-only quaternion slerp.
- Create `src/Validator_pkg/validator_inference/src/validator_inference/model_environment.py`: manifest schema, paths, state, fingerprints, preflight, and sanitized process environment.
- Create `src/Validator_pkg/validator_inference/src/validator_inference/environment_setup.py`: candidate lifecycle, `uv` execution, ROS overlay, host checks, verification, and pruning.
- Create `src/Validator_pkg/validator_inference/src/validator_inference/ros_compatibility.py`: target/system peer used by cross-version ROS smoke tests.
- Create `src/Validator_pkg/validator_inference/scripts/model_env_exec`: launch-prefix executable that replaces itself with the selected interpreter.
- Create `src/Validator_pkg/validator_inference/scripts/openloop_cli.py`: setup, run, verify, prune, bag, and shorthand command controller.
- Modify `src/Validator_pkg/validator_inference/src/validator_inference/input_buffer.py`: use the local slerp and remove the Python-3.8-only TF extension path.
- Modify `src/Validator_pkg/validator_inference/launch/openloop_inference.launch`: prefix only `inference_engine_node.py`.
- Modify `src/Validator_pkg/validator_pipeline/launch/openloop_validator.launch`: forward the absolute model environment root.
- Modify `src/Validator_pkg/validator_inference/CMakeLists.txt` and `package.xml`: install executables, manifests, and locks.
- Modify `src/Validator_pkg/config/profile.json`: pin the InterFuser checkpoint provenance/hash and declare the ResNet50D cache asset.
- Modify `.gitignore`, `README.md`, and `docs/openloop_validator.md`: ignore local environments and document canonical commands and diagnostics.
- Create tests `test_quaternion_math.py`, `test_model_environment.py`, `test_environment_setup.py`, `test_model_env_exec.py`, `test_openloop_cli.py`, and `test_ros_compatibility.py`; extend existing launch, asset, install, and smoke contracts.

### Task 1: Remove the Python-3.8 TF Binary Dependency

**Files:**
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/quaternion_math.py`
- Modify: `src/Validator_pkg/validator_inference/src/validator_inference/input_buffer.py`
- Create: `src/Validator_pkg/validator_inference/test/test_quaternion_math.py`
- Modify: `src/Validator_pkg/validator_inference/test/test_input_buffer.py`

- [ ] **Step 1: Add numerical and import-isolation tests**

Add tests with these concrete cases:

```python
def test_slerp_normalizes_endpoints():
    assert quaternion_slerp([0, 0, 0, 2], [0, 0, 2, 0], 0.0) == pytest.approx([0, 0, 0, 1])
    assert quaternion_slerp([0, 0, 0, 2], [0, 0, 2, 0], 1.0) == pytest.approx([0, 0, 1, 0])


def test_slerp_uses_shortest_path():
    result = quaternion_slerp([0, 0, 0, 1], [0, 0, -1, 0], 0.5)
    assert abs(float(np.dot(result, [0, 0, 2 ** -0.5, 2 ** -0.5]))) == pytest.approx(1.0)


def test_importing_input_buffer_does_not_load_tf(monkeypatch):
    for name in tuple(sys.modules):
        if name == "tf" or name.startswith("tf.") or name.startswith("tf2_py"):
            sys.modules.pop(name)
    importlib.reload(input_buffer)
    assert not any(name == "tf" or name.startswith("tf.") or name.startswith("tf2_py") for name in sys.modules)
```

- [ ] **Step 2: Prove the tests fail before implementation**

Run:

```bash
bash -lc 'source /opt/ros/noetic/setup.bash && source devel/setup.bash && /usr/bin/python3 -m pytest src/Validator_pkg/validator_inference/test/test_quaternion_math.py src/Validator_pkg/validator_inference/test/test_input_buffer.py -q'
```

Expected: collection fails because `validator_inference.quaternion_math` does not exist.

- [ ] **Step 3: Implement normalized shortest-path slerp**

Implement `quaternion_slerp(q0, q1, fraction)` with `numpy.asarray`, zero-norm rejection, endpoint normalization, sign reversal when the dot product is negative, normalized linear interpolation when `dot > 0.9995`, and the standard sine-weighted interpolation otherwise. Give the module and function Korean docstrings containing the repository-required input/output contract terms. Replace the conditional `tf.transformations` import in `input_buffer.py` with:

```python
from validator_inference.quaternion_math import quaternion_slerp
```

- [ ] **Step 4: Run focused and convention tests**

Run the Step 2 command, then:

```bash
bash -lc 'source /opt/ros/noetic/setup.bash && source devel/setup.bash && /usr/bin/python3 -m pytest src/Validator_pkg/validator_inference/test/test_comment_conventions.py -q'
```

Expected: both commands pass.

- [ ] **Step 5: Commit the isolated compatibility fix**

```bash
git add src/Validator_pkg/validator_inference/src/validator_inference/quaternion_math.py src/Validator_pkg/validator_inference/src/validator_inference/input_buffer.py src/Validator_pkg/validator_inference/test/test_quaternion_math.py src/Validator_pkg/validator_inference/test/test_input_buffer.py
git commit -m "fix: remove binary tf dependency from inference"
```

### Task 2: Define and Validate the Four Environment Contracts

**Files:**
- Create: `src/Validator_pkg/validator_inference/config/model_environments.json`
- Create: `src/Validator_pkg/validator_inference/requirements/interfuser.in`
- Create: `src/Validator_pkg/validator_inference/requirements/tfv6.in`
- Create: `src/Validator_pkg/validator_inference/requirements/diffusiondrive_v2.in`
- Create: `src/Validator_pkg/validator_inference/requirements/sparsedrive_v2.in`
- Create: `src/Validator_pkg/validator_inference/requirements/interfuser.lock`
- Create: `src/Validator_pkg/validator_inference/requirements/tfv6.lock`
- Create: `src/Validator_pkg/validator_inference/requirements/diffusiondrive_v2.lock`
- Create: `src/Validator_pkg/validator_inference/requirements/sparsedrive_v2.lock`
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/model_environment.py`
- Create: `src/Validator_pkg/validator_inference/test/test_model_environment.py`

- [ ] **Step 1: Write schema and exact-pin tests**

The tests must assert the exact model set and primary pins:

```python
EXPECTED = {
    "interfuser": ("3.8.20", "torch==2.4.1+cu121", "torchvision==0.19.1+cu121", "recommended"),
    "tfv6": ("3.10.20", "torch==2.5.0+cu124", "torchvision==0.20.0+cu124", "required"),
    "diffusiondrive_v2": ("3.9.25", "torch==2.0.1+cu118", "torchvision==0.15.2+cu118", "recommended"),
    "sparsedrive_v2": ("3.9.25", "torch==2.0.1+cu118", "torchvision==0.15.2+cu118", "extension_required"),
}


def test_manifest_defines_exact_supported_models(manifest):
    assert set(manifest.models) == set(EXPECTED)
    for model_id, (python, torch, torchvision, cuda_policy) in EXPECTED.items():
        definition = manifest.models[model_id]
        assert definition.python_version == python
        assert definition.cuda_policy == cuda_policy
        requirement_text = definition.requirements_input.read_text()
        assert torch in requirement_text
        assert torchvision in requirement_text
```

Also test unknown model rejection, unknown keys, invalid SHA values, invalid CUDA policy, absolute/escaping paths, exact `uv_version == "0.11.14"`, platform `x86_64-manylinux_2_31`, ordered official PyTorch/PyPI indexes, and lock/input existence.

- [ ] **Step 2: Verify red state**

Run:

```bash
/usr/bin/python3 -m pytest src/Validator_pkg/validator_inference/test/test_model_environment.py -q
```

Expected: import fails because `model_environment.py` and its manifest do not exist.

- [ ] **Step 3: Implement typed manifest loading**

Define immutable `EnvironmentManifest`, `ModelEnvironmentDefinition`, `EnvironmentPaths`, `AssetFingerprint`, `EnvironmentState`, and `ActiveEnvironmentCheck` dataclasses. Expose `load_environment_manifest(path: Path) -> EnvironmentManifest`, `select_model_environment(manifest, model_id) -> ModelEnvironmentDefinition`, `environment_paths(project_root, model_id) -> EnvironmentPaths`, `sha256_file(path) -> str`, `build_uv_venv_argv(uv, definition, candidate) -> tuple[str, ...]`, and `build_uv_sync_argv(uv, definition, candidate) -> tuple[str, ...]`. Raise `EnvironmentConfigError` for invalid tracked definitions and `EnvironmentStateError` for malformed local state.

Construct argv arrays, never shell strings. Venv creation is `uv venv --no-project --managed-python --python <exact-patch> <candidate>`. Sync is `uv pip sync --python <candidate>/bin/python --require-hashes --strict --no-python-downloads <lock>`; a compiled lock already contains the full dependency resolution, and pinned `uv 0.11.14` does not expose `--no-deps` on `pip sync`.

- [ ] **Step 4: Add all four inputs and generated locks**

Every `.in` includes exact torch/torchvision pins and target-Python builds of `PyYAML`, `rospkg`, `catkin_pkg`, `netifaces`, and `python-dateutil`. TFv6 also pins torchaudio 2.5.0+cu124, NumPy 1.26, timm 1.0.19, transformers 4.46.3, and torchmetrics 0.11; DiffusionDrive pins protobuf 4.25.3, diffusers, and einops; SparseDrive pins protobuf 3.20.* and its upstream direct dependencies. DiffusionDrive and SparseDrive replace the unhashable upstream Git reference with `nuplan-devkit @ https://github.com/motional/nuplan-devkit/archive/ce3c323af01c0d7ec5672f7832ef53f9c679aab0.tar.gz`; `uv pip compile --generate-hashes` must record archive SHA256 `5ab521c1af439902fd8e3df2619da75a03ade5814b35cd37c4c920fd45455e62`. Generate each lock with the manifest-recorded `uv 0.11.14`, Python patch, `--python-platform x86_64-manylinux_2_31`, official CUDA wheel index, PyPI secondary index, `--generate-hashes`, and fixed `--exclude-newer` timestamp. Check the generated lock contains hashes for every installable line.

- [ ] **Step 5: Pass manifest and lock tests**

Run the Step 2 command and:

```bash
/home/stier/.local/bin/uv pip compile --version
```

Expected: tests pass and the version output contains `uv 0.11.14`.

- [ ] **Step 6: Commit environment definitions**

```bash
git add src/Validator_pkg/validator_inference/config/model_environments.json src/Validator_pkg/validator_inference/requirements src/Validator_pkg/validator_inference/src/validator_inference/model_environment.py src/Validator_pkg/validator_inference/test/test_model_environment.py
git commit -m "feat: define isolated model environments"
```

### Task 3: Add State, Fingerprints, and Read-only Preflight

**Files:**
- Modify: `src/Validator_pkg/validator_inference/src/validator_inference/model_environment.py`
- Modify: `src/Validator_pkg/validator_inference/test/test_model_environment.py`

- [ ] **Step 1: Add stale-state and environment-sanitization tests**

Test that `inspect_active_environment` rejects a missing `current`, non-symlink `current`, missing interpreter, wrong model/Python/schema, changed manifest/lock/overlay hashes, stale source commit, and changed asset size or `mtime_ns`. Test that an unchanged fingerprint avoids calling `sha256_file`. Test that process environment drops inherited `PYTHONPATH`, `PYTHONHOME`, `PYTHONUSERBASE`, and foreign `VIRTUAL_ENV`, while preserving ROS master variables and setting exactly the selected `PATH`, `VIRTUAL_ENV`, `PYTHONNOUSERSITE=1`, overlay/Catkin `PYTHONPATH`, and repository `TORCH_HOME`.

- [ ] **Step 2: Prove the new contracts fail**

Run:

```bash
/usr/bin/python3 -m pytest src/Validator_pkg/validator_inference/test/test_model_environment.py -q
```

Expected: failures name missing `inspect_active_environment` and `build_model_process_environment`.

- [ ] **Step 3: Implement deterministic state serialization and inspection**

Add `read_environment_state(environment_dir) -> EnvironmentState`, `write_environment_state(path, state) -> None`, `inspect_active_environment(paths, definition, expected) -> ActiveEnvironmentCheck`, and `build_model_process_environment(inherited, environment_dir, ros_overlay, catkin_python_path, project_root) -> dict[str, str]`.

Serialize JSON with `sort_keys=True`, reject unknown fields, resolve the selected build beneath `builds/`, and return one actionable reason rather than raising for ordinary missing/stale state. Use only size and nanosecond mtime for run-time asset checks; deep SHA stays in setup/verify.

- [ ] **Step 4: Run focused tests and commit**

Run the Step 2 command. Expected: pass.

```bash
git add src/Validator_pkg/validator_inference/src/validator_inference/model_environment.py src/Validator_pkg/validator_inference/test/test_model_environment.py
git commit -m "feat: validate active model environment state"
```

### Task 4: Build Candidates and the Pure-Python ROS Overlay

**Files:**
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/environment_setup.py`
- Create: `src/Validator_pkg/validator_inference/test/test_environment_setup.py`

- [ ] **Step 1: Add candidate and overlay security tests**

Use temporary fake ROS package trees to test this exact allowlist:

```python
PURE_ROS_PACKAGES = (
    "rospy", "rosgraph", "roslib", "genpy", "genmsg", "catkin", "roscpp",
    "rosgraph_msgs", "std_msgs", "sensor_msgs", "geometry_msgs", "nav_msgs",
    "tf2_msgs", "actionlib_msgs",
)

```

Exercise `create_candidate(paths, build_id) -> Path`, `build_ros_overlay(source_root, destination, packages=PURE_ROS_PACKAGES) -> RosOverlayManifest`, `verify_ros_overlay(destination, manifest_path) -> str`, `activate_candidate(paths, candidate) -> None`, and `prune_inactive_builds(paths) -> tuple[Path, ...]`.

Assert only declared `.py` files are copied, symlinks are not followed, `.so` files and missing import closure fail, every copied file has source path and SHA256, extra/modified files fail verification, a failed candidate preserves the old `current`, activation uses a relative link, and prune never deletes the active build.

- [ ] **Step 2: Verify red state**

Run:

```bash
/usr/bin/python3 -m pytest src/Validator_pkg/validator_inference/test/test_environment_setup.py -q
```

Expected: import fails because `environment_setup.py` does not exist.

- [ ] **Step 3: Implement candidate and overlay operations**

Create candidates directly at their unique final paths so venv-generated scripts never move. Walk only allowlisted packages with `os.scandir`, reject symlinks and native extensions, ignore `.pyc` and `__pycache__`, copy with `shutil.copy2`, and write a sorted manifest atomically using a sibling temporary file plus `os.replace`. Activate by creating a temporary relative symlink under the model root and replacing `current` with `os.replace` only after all validation passes.

- [ ] **Step 4: Add host prerequisite phase tests**

Inject command lookup/runner/GPU probes. Assert unsupported OS, wrong `uv`, missing ROS, missing CUDA for TFv6, missing `nvcc 11.8`/`CUDA_HOME`/compiler/extensions or null three-camera calibration for SparseDrive fail before any downloader or venv call. SparseDrive uses cwd `models/sources/sparsedrive_v2/navsim/agents/sparsedrive/ops` and executes the candidate Python on `setup.py build_ext --inplace` with `CUDA_HOME=/usr/local/cuda-11.8`, `FORCE_CUDA=1`, `TORCH_CUDA_ARCH_LIST=8.6`, `CC=/usr/bin/gcc`, and `CXX=/usr/bin/g++`; verify imports `deformable_aggregation_ext` and `deformable_aggregation_with_depth_ext` under that package. Assert InterFuser and DiffusionDrive return a recorded warning without CUDA, and exact `uv` argv from Task 2 is executed with `check=True`.

- [ ] **Step 5: Implement setup phases and deep verification**

Expose `setup_environment(project_root, model_id, recreate=False, runner=subprocess.run)`, `verify_environment(project_root, model_id)`, and `prune_environment_builds(project_root, model_id)`. Execute phases in the design order, record the failed phase in `EnvironmentSetupError`, deep-hash sources/assets during setup/verify, write state last, and switch `current` only after all required checks pass.

- [ ] **Step 6: Run tests and commit**

Run the Step 2 command plus `test_model_environment.py`. Expected: pass.

```bash
git add src/Validator_pkg/validator_inference/src/validator_inference/environment_setup.py src/Validator_pkg/validator_inference/test/test_environment_setup.py
git commit -m "feat: create atomic model environments"
```

### Task 5: Add the Inference-only Interpreter Boundary

**Files:**
- Create: `src/Validator_pkg/validator_inference/scripts/model_env_exec`
- Create: `src/Validator_pkg/validator_inference/test/test_model_env_exec.py`
- Modify: `src/Validator_pkg/validator_inference/launch/openloop_inference.launch`
- Modify: `src/Validator_pkg/validator_pipeline/launch/openloop_validator.launch`
- Modify: `src/Validator_pkg/validator_inference/test/test_layer_integration.py`
- Modify: `src/Validator_pkg/validator_inference/test/test_openloop_smoke.py`

- [ ] **Step 1: Write wrapper exec and launch XML tests**

Run the executable in subprocesses with a fake venv Python and fake Catkin devel/install node paths. Assert it preserves every ROS remapping argument, propagates child exit code, rejects relative environment roots/model IDs with path separators, derives `<prefix>/lib/python3/dist-packages`, selects `.venvs/MODEL/current/bin/python`, and provides the sanitized variables from Task 3. Parse both launch XML files and assert exactly one node has `launch-prefix`, that node is `inference_engine_node.py`, and `model_env_root` is forwarded to the included inference launch.

- [ ] **Step 2: Prove wrapper/launch tests fail**

Run:

```bash
bash -lc 'source /opt/ros/noetic/setup.bash && source devel/setup.bash && /usr/bin/python3 -m pytest src/Validator_pkg/validator_inference/test/test_model_env_exec.py src/Validator_pkg/validator_inference/test/test_openloop_smoke.py src/Validator_pkg/validator_inference/test/test_layer_integration.py -q'
```

Expected: failures report missing wrapper and launch arguments.

- [ ] **Step 3: Implement the exec boundary**

Use `#!/usr/bin/python3` and standard library startup. Read absolute `OPENLOOP_MODEL_ENV_ROOT`, `OPENLOOP_MODEL_ID`, and `OPENLOOP_PROJECT_ROOT`; treat `sys.argv[1]` as the Catkin-generated node relay; derive the active prefix without resolving the relay back to source; import `model_environment` from the active Catkin path; then call:

```python
os.execve(
    str(venv_python),
    [str(venv_python), node_executable, *sys.argv[2:]],
    child_environment,
)
```

Make the wrapper executable. In `openloop_inference.launch`, add absolute environment root/project root args and node-local `<env>` entries, and apply `$(find validator_inference)/scripts/model_env_exec` only as the engine launch prefix. Forward args through the pipeline launch.

- [ ] **Step 4: Pass focused tests and commit**

Run the Step 2 command. Expected: pass.

```bash
git add src/Validator_pkg/validator_inference/scripts/model_env_exec src/Validator_pkg/validator_inference/launch/openloop_inference.launch src/Validator_pkg/validator_pipeline/launch/openloop_validator.launch src/Validator_pkg/validator_inference/test/test_model_env_exec.py src/Validator_pkg/validator_inference/test/test_layer_integration.py src/Validator_pkg/validator_inference/test/test_openloop_smoke.py
git commit -m "feat: launch inference in selected model environment"
```

### Task 6: Add the Repository-local `openloop` Command

**Files:**
- Create: `openloop`
- Create: `src/Validator_pkg/validator_inference/scripts/openloop_cli.py`
- Create: `src/Validator_pkg/validator_inference/test/test_openloop_cli.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add parser, argv, and no-mutation tests**

Load the script module and test `setup MODEL [--recreate]`, `run MODEL [--rviz]`, shorthand `MODEL [--rviz]`, `verify MODEL`, `prune MODEL`, and `bag [--path PATH]`. Assert unknown models/options exit 2, bag checks existence, run never invokes setup/download, stale run prints exactly one recovery command, and subprocess argv is an array. The launch contract is:

```python
[
    "roslaunch", "Validator_pkg", "openloop_validator.launch",
    f"model_id:={model_id}",
    f"model_env_root:={project_root / '.venvs'}",
    f"show_rviz:={'true' if rviz else 'false'}",
]
```

The bag contract is `rosbag play --clock <absolute-bag-path>`. Test the root shim with no `devel/`: it must source `/opt/ros/noetic/setup.bash`, conditionally source devel when present, prepend the source package path, and exec `/usr/bin/python3 scripts/openloop_cli.py` without changing argument boundaries.

- [ ] **Step 2: Verify red state**

Run:

```bash
/usr/bin/python3 -m pytest src/Validator_pkg/validator_inference/test/test_openloop_cli.py -q
```

Expected: scripts are missing.

- [ ] **Step 3: Implement CLI dispatch**

Keep `_parser`, `_normalize_argv`, `_launch_argv`, `_bag_argv`, and `main` importable without ROS messages. Use `os.execvpe` for long-lived roslaunch/rosbag commands so Ctrl-C and exit status are unchanged. Use `environment_setup` APIs for setup/verify/prune. Resolve the default bag from the repository profile configuration, print concise phase errors to stderr, and return nonzero before ROS on preflight failure.

- [ ] **Step 4: Implement clean-checkout bootstrap and ignore local builds**

The Bash shim computes `PROJECT_ROOT` from its own path and executes this exact flow; add `/.venvs/` to `.gitignore`:

```bash
#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/noetic/setup.bash
if [[ -f "$PROJECT_ROOT/devel/setup.bash" ]]; then
  source "$PROJECT_ROOT/devel/setup.bash"
fi
export PYTHONPATH="$PROJECT_ROOT/src/Validator_pkg/validator_inference/src${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 "$PROJECT_ROOT/src/Validator_pkg/validator_inference/scripts/openloop_cli.py" "$@"
```

- [ ] **Step 5: Pass CLI tests and commit**

Run the Step 2 command and `bash -n openloop`. Expected: pass.

```bash
git add openloop .gitignore src/Validator_pkg/validator_inference/scripts/openloop_cli.py src/Validator_pkg/validator_inference/test/test_openloop_cli.py
git commit -m "feat: add simple openloop command"
```

### Task 7: Make InterFuser Assets Deterministic

**Files:**
- Modify: `src/Validator_pkg/config/profile.json`
- Modify: `src/Validator_pkg/validator_inference/test/test_assets_and_model_selection.py`
- Modify: `src/Validator_pkg/validator_inference/test/test_profile_store.py`
- Modify: `src/Validator_pkg/validator_inference/src/validator_inference/environment_setup.py`

- [ ] **Step 1: Add provenance and cache-asset tests**

Assert the InterFuser runtime assets contain exactly the checkpoint and backbone cache entry. Assert checkpoint SHA256 `39548dcfa56cdc06b8ccc52d7f6187a7d70ef36b05b7027db4d64a57ac3e30b2`, Drivora commit `25d56d2300f422d502c6549695b9e49efd5e5e00`, Google Drive ID `1GKiASmGPbD4FwHkUoVfGRk_lMLrGb2f6`, and an explicit third-party compatibility label. Assert the backbone path is `models/shared/torch-cache/hub/checkpoints/resnet50d_ra2-464e36ba.pth` with SHA256 `464e36baa940a80971e717454e61e8f6fc1a1b284b7a3dbb6f6fd4c3da78533d`. Assert the dead URL is absent.

- [ ] **Step 2: Run tests to observe the old profile failure**

Run:

```bash
bash -lc 'source /opt/ros/noetic/setup.bash && source devel/setup.bash && /usr/bin/python3 -m pytest src/Validator_pkg/validator_inference/test/test_assets_and_model_selection.py src/Validator_pkg/validator_inference/test/test_profile_store.py -q'
```

Expected: the profile lacks both fixed hashes and provenance.

- [ ] **Step 3: Update profile and local-asset setup behavior**

Record both assets and their hashes. Setup may reuse matching local files and may copy the existing verified user cache file into the declared repository cache path, but must not label or silently download the third-party checkpoint as official. If absent, report the exact expected path, hash, and provenance. Ensure environment state records both fingerprints and `TORCH_HOME` points to `models/shared/torch-cache`.

- [ ] **Step 4: Pass tests and commit**

Run the Step 2 command plus `test_environment_setup.py`. Expected: pass.

```bash
git add src/Validator_pkg/config/profile.json src/Validator_pkg/validator_inference/src/validator_inference/environment_setup.py src/Validator_pkg/validator_inference/test/test_assets_and_model_selection.py src/Validator_pkg/validator_inference/test/test_profile_store.py
git commit -m "fix: pin InterFuser runtime assets"
```

### Task 8: Install Every Runtime Component and Update Operator Documentation

**Files:**
- Modify: `src/Validator_pkg/validator_inference/CMakeLists.txt`
- Modify: `src/Validator_pkg/validator_inference/package.xml`
- Modify: `src/Validator_pkg/validator_inference/setup.py`
- Modify: `src/Validator_pkg/validator_inference/test/test_install_contract.py`
- Modify: `src/Validator_pkg/validator_inference/test/test_package_layout.py`
- Modify: `src/Validator_pkg/validator_inference/test/test_comment_conventions.py`
- Modify: `src/Validator_pkg/validator_inference/test/test_openloop_smoke.py`
- Modify: `README.md`
- Modify: `docs/openloop_validator.md`

- [ ] **Step 1: Add install/package/document contract tests**

Assert `openloop_cli.py` is installed with `catkin_install_python`; `model_env_exec` is installed unchanged under `${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts`; launch, environment JSON, requirements inputs, and locks are package data; all production `.py` files satisfy Korean docstring conventions; and operator docs contain only the canonical workflow:

```bash
./openloop setup interfuser
./openloop interfuser --rviz
./openloop bag
```

Document `verify`, `prune`, direct advanced launch with absolute `model_env_root`, model readiness scope, third-party checkpoint provenance, and actionable setup diagnostics.

- [ ] **Step 2: Prove install/docs contracts fail**

Run:

```bash
bash -lc 'source /opt/ros/noetic/setup.bash && source devel/setup.bash && /usr/bin/python3 -m pytest src/Validator_pkg/validator_inference/test/test_install_contract.py src/Validator_pkg/validator_inference/test/test_package_layout.py src/Validator_pkg/validator_inference/test/test_comment_conventions.py src/Validator_pkg/validator_inference/test/test_openloop_smoke.py -q'
```

Expected: missing install rules and old operator command assertions fail.

- [ ] **Step 3: Implement install rules and docs**

Use `install(PROGRAMS scripts/model_env_exec DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts)` for the wrapper so CMake does not rewrite its interpreter, `install(DIRECTORY requirements/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/requirements)` for lock data, and `install(FILES config/model_environments.json DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config)` for the manifest. Keep source-tree CLI bootstrap independent of install space.

- [ ] **Step 4: Build, run tests, and commit**

Run the Step 2 command and:

```bash
bash -lc 'source /opt/ros/noetic/setup.bash && catkin_make'
```

Expected: build and tests pass.

```bash
git add src/Validator_pkg/validator_inference/CMakeLists.txt src/Validator_pkg/validator_inference/package.xml src/Validator_pkg/validator_inference/setup.py src/Validator_pkg/validator_inference/test README.md docs/openloop_validator.md
git commit -m "docs: install and document openloop environments"
```

### Task 9: Verify Cross-version ROS Communication and Process Semantics

**Files:**
- Create: `src/Validator_pkg/validator_inference/src/validator_inference/ros_compatibility.py`
- Create: `src/Validator_pkg/validator_inference/test/test_ros_compatibility.py`
- Modify: `src/Validator_pkg/validator_inference/src/validator_inference/environment_setup.py`

- [ ] **Step 1: Add origin-audit and cleanup unit tests**

Define `RuntimePathPolicy`, `RosSmokeReport`, `validate_loaded_module_origins`, and `run_ros_compatibility_smoke`. Unit tests feed fake module origins and require rejection of CPython-3.8 extensions under `/opt/ros`, ROS packages outside `.ros-overlay`, project messages outside the active Catkin path, and arbitrary inherited venvs. Force peer startup failure and assert temporary roscore and every process group terminate in `finally`.

- [ ] **Step 2: Add parametrized 3.8/3.9/3.10 integration cases**

For CPython 3.8.20, 3.9.25, and 3.10.20, build a minimal target venv/overlay independent of model assets and exchange both directions:

```text
sensor_msgs/Image
sensor_msgs/PointCloud2
validator_msgs/DrivingFeature
tf2_msgs/TFMessage
nav_msgs/Path
validator_msgs/InferenceMetric
rosgraph_msgs/Clock
```

Verify `/use_sim_time`, clock advancement, inference imports, selected `sys.executable`, exact target `PYTHONPATH`, and loaded extension origins. Mark network/interpreter creation tests `ros_compatibility` so the default suite stays offline.

- [ ] **Step 3: Implement the system/target peer and bounded orchestration**

`ros_compatibility.py --role system|target --result PATH` initializes anonymous nodes, publishes/subscribes the typed fixtures, records received fields as JSON, and exits nonzero on mismatch. `run_ros_compatibility_smoke` reserves a localhost port, starts `roscore`, waits with a deadline, launches both peers in process groups, collects their results, and terminates all children on success, timeout, or exception.

- [ ] **Step 4: Add install-prefix and signal integration tests**

Build a temporary Catkin install prefix and prove the wrapper derives its installed `lib/python3/dist-packages`. Launch a prefixed stub with real `roslaunch`, send SIGINT, and verify ROS remapping arguments, child status, required-node shutdown, and no surviving wrapper/child process.

- [ ] **Step 5: Run unit and opted-in integration checks**

Run:

```bash
bash -lc 'source /opt/ros/noetic/setup.bash && source devel/setup.bash && /usr/bin/python3 -m pytest src/Validator_pkg/validator_inference/test/test_ros_compatibility.py -q'
bash -lc 'source /opt/ros/noetic/setup.bash && source devel/setup.bash && VALIDATOR_RUN_ROS_COMPAT=1 /usr/bin/python3 -m pytest src/Validator_pkg/validator_inference/test/test_ros_compatibility.py -m ros_compatibility -vv'
```

Expected: all unit and all three cross-version cases pass.

- [ ] **Step 6: Commit compatibility validation**

```bash
git add src/Validator_pkg/validator_inference/src/validator_inference/ros_compatibility.py src/Validator_pkg/validator_inference/src/validator_inference/environment_setup.py src/Validator_pkg/validator_inference/test/test_ros_compatibility.py
git commit -m "test: validate cross-version ROS model runtimes"
```

### Task 10: Complete InterFuser Setup and End-to-end Bag Acceptance

**Files:**
- Modify only when a test exposes a defect in files introduced above.

- [ ] **Step 1: Run the complete network-free regression suite**

```bash
bash -lc 'source /opt/ros/noetic/setup.bash && source devel/setup.bash && /usr/bin/python3 -m pytest src/Validator_pkg/validator_inference/test src/Validator_pkg/validator_msgs/test -q'
bash -lc 'source /opt/ros/noetic/setup.bash && catkin_make'
```

Expected: all tests pass and Catkin build succeeds.

- [ ] **Step 2: Create and deeply verify the repository-owned InterFuser environment**

```bash
./openloop setup interfuser
./openloop verify interfuser
```

Expected: `.venvs/interfuser/current/bin/python` reports Python 3.8.20, torch 2.4.1+cu121 sees the RTX 3090, both runtime assets match their SHA256 values, strict model load passes, the ROS smoke passes, and state contains no external project venv path.

- [ ] **Step 3: Validate all other model preflight definitions without claiming installation**

Run unit-level preflight probes for `tfv6`, `diffusiondrive_v2`, and `sparsedrive_v2`. Expected: TFv6 reports absent assets or its required CUDA runtime precisely; DiffusionDrive reports missing declared assets while its CUDA recommendation remains a warning; SparseDrive reports complete camera calibration and `nvcc 11.8` blockers before download.

- [ ] **Step 4: Run a 20-second real rosbag acceptance**

Start `./openloop interfuser` without RViz, then in another process start `./openloop bag`; stop after at least 20 seconds of simulated time. Assert the inference engine stays alive, `/waypoints` publishes, and the new `records/openloop/inference_*_interfuser.csv` contains an `inference_completed` row. Capture `sys.executable`, CUDA device, and sanitized `PYTHONPATH` from the engine log and verify they point only to `.venvs/interfuser/current`, its overlay, and this workspace's active Catkin prefix.

- [ ] **Step 5: Exercise the user-facing RViz command and stale-state diagnostic**

Run `./openloop interfuser --rviz`, confirm launch reaches the waiting-for-bag state, then stop it with Ctrl-C. In a temporary copied state fixture alter an asset fingerprint and assert run exits before ROS with:

```text
interfuser environment is missing or stale.
Run: ./openloop setup interfuser
```

- [ ] **Step 6: Inspect final diff and commit acceptance fixes**

```bash
git status --short
git diff --check
git log --oneline --decorate -12
```

Expected: no untracked environment assets, no whitespace errors, and only scoped feature changes. Commit any defect fixes with a message describing the behavior corrected; do not commit `.venvs`, model binaries, bags, logs, or records.
