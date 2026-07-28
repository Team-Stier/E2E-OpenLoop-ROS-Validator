# BEV Shell Command Implementation Plan

> **For agentic workers:** Use inline execution and do not commit or push.

**Goal:** Make the command `bev` launch the local LiDAR attitude RViz and looping rosbag safely.

**Architecture:** Keep `~/.bashrc` as a thin function and place lifecycle logic in one repository script with preflight checks and signal cleanup.

**Tech Stack:** Bash, ROS1 Noetic command-line tools, catkin workspace.

## Global Constraints

- Preserve existing `.bashrc` contents.
- Do not commit or push.
- Reuse an existing ROS master and never terminate a master the script did not start.
- Terminate only processes started by the current invocation.

### Task 1: Executable launcher

**Files:**
- Create: `scripts/run_lidar_bev_debug.sh`
- Create: `src/Validator_pkg/validator_pipeline/test/test_bev_shell_command.py`

- [ ] Write a failing contract test for syntax, paths, preflight, cleanup trap, launch, and looping bag arguments.
- [ ] Run the test and verify it fails because the script is missing.
- [ ] Implement the smallest launcher satisfying the contract.
- [ ] Run the focused test and Bash syntax check.

### Task 2: Bash function installation

**Files:**
- Modify locally: `/home/stier/.bashrc`

- [ ] Copy `.bashrc` to a writable temporary file and record its checksum.
- [ ] Patch the temporary copy with one idempotent marker block.
- [ ] Verify the original checksum has not changed.
- [ ] Install the patched copy with explicit approval.
- [ ] Start a clean Bash process and verify `type bev` resolves to the exact function.

### Task 3: Runtime verification

- [ ] Stop only the earlier diagnostic RViz/bag processes started in this session.
- [ ] Run `bev` in a user terminal.
- [ ] Confirm the diagnostic node, RViz, `/clock`, and all three debug topics.
- [ ] Confirm output cadence and report how to stop and restart.
