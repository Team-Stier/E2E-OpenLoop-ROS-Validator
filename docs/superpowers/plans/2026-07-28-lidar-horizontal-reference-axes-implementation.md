# LiDAR Nominal and Current Axes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare nominal LiDAR XY axes against current LiDAR-estimated XY directions without translucent floor planes.

**Architecture:** Derive plane-parallel X/Y directions from the confidence-gated RANSAC normal. Publish only nominal marker ID `5` and current marker ID `6` on the existing marker topic.

**Tech Stack:** Python 3.8, NumPy, ROS1 Noetic `Marker.LINE_LIST`, catkin nosetests.

## Global Constraints

- Common origin: configured LiDAR translation `(2.0, 0.0, 1.5)`.
- ID `5`: yellow nominal axes, width `0.15 m`.
- ID `6`: red current axes, width `0.12 m`, valid estimate only.
- Remove translucent blue floor ID `1` and translucent red floor ID `7`.
- Do not publish height line ID `8` or height text ID `9`.
- Preserve point coloring, status text, normal arrows, topics, and playback.
- Do not commit or push.

---

### Task 1: Verify plane-parallel axis directions

- [ ] Add a failing test for `plane_axis_directions(normal)`.
- [ ] Implement normalized X/Y slice directions perpendicular to the normal.
- [ ] Run focused and complete LiDAR tests.

### Task 2: Publish axes without floor planes

- [ ] Assert IDs `5` and `6` are present for a valid estimate.
- [ ] Assert IDs `1`, `7`, `8`, and `9` are absent.
- [ ] Remove blue and red translucent plane marker blocks.
- [ ] Remove unused fitted-plane and height geometry helpers.
- [ ] Run focused and complete tests.

### Task 3: Build and live-verify

- [ ] Run Python compilation, `git diff --check`, catkin build, and all tests.
- [ ] Restart only the owned `bev` process group.
- [ ] Verify live IDs `5` and `6`, and absence of IDs `1`, `7`, `8`, and `9`.
- [ ] Confirm one diagnostic node, one RViz node, and one rosbag playback node.
