# horizontal_pkg README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a concise, source-verified package README for offline IMU analysis, LiDAR horizontalization, and comparison animation generation.

**Architecture:** The README is a package-local entrypoint that documents current Python CLI contracts without adding runtime behavior. Topic/frame tables, a compact flow, reproducible commands, asset links, and limitations are derived directly from the existing modules.

**Tech Stack:** Markdown, ROS1 Noetic, rosbag, Python 3.

## Global Constraints

- Modify files only under `src/Validator_pkg/horizontal_pkg`.
- Do not change ROS interfaces or runtime code.
- Document only values confirmed in the current implementation.
- Keep the README short enough to scan quickly.

---

### Task 1: Add and verify the package README

**Files:**
- Create: `src/Validator_pkg/horizontal_pkg/README.md`
- Verify: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/lidar_horizontal.py`
- Verify: `src/Validator_pkg/horizontal_pkg/src/horizontal_pkg/lidar_comparison_animation.py`

**Interfaces:**
- Consumes: existing CLI defaults, topic names, frame names, and documentation assets.
- Produces: package onboarding documentation only.

- [ ] **Step 1: Write README** with purpose, topic/frame tables, offline flow, two commands, asset links, and limitations.
- [ ] **Step 2: Verify constants** using `rg` against the two Python modules.
- [ ] **Step 3: Verify links** by checking every referenced package-relative file exists.
- [ ] **Step 4: Run checks** with `git diff --check` and the validator package-layout focused test.
- [ ] **Step 5: Commit** with `docs: add horizontal package README`.
