#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
BAG_PATH="$REPOSITORY_ROOT/rosbag/2026-07-16-17-13-14.bag"
ROSCORE_PID=""
ROSLAUNCH_PID=""
ROSCORE_LOG="/tmp/team_stier_bev_roscore_$$.log"
ROSLAUNCH_LOG="/tmp/team_stier_bev_roslaunch_$$.log"

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM HUP
  set +e

  if [[ -n "$ROSLAUNCH_PID" ]] && kill -0 "$ROSLAUNCH_PID" 2>/dev/null; then
    kill -INT "$ROSLAUNCH_PID" 2>/dev/null
    wait "$ROSLAUNCH_PID" 2>/dev/null
  fi

  if [[ -n "$ROSCORE_PID" ]] && kill -0 "$ROSCORE_PID" 2>/dev/null; then
    kill -INT "$ROSCORE_PID" 2>/dev/null
    wait "$ROSCORE_PID" 2>/dev/null
  fi

  exit "$exit_code"
}
trap cleanup EXIT INT TERM HUP

fail() {
  echo "[bev] 오류: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "필요한 파일이 없습니다: $1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "명령을 찾을 수 없습니다: $1"
}

wait_for_master() {
  local attempt
  for attempt in {1..50}; do
    if rosnode list >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

wait_for_debugger() {
  local attempt
  for attempt in {1..100}; do
    if rosnode list 2>/dev/null | grep -Fxq "/lidar_attitude_debugger"; then
      return 0
    fi
    if ! kill -0 "$ROSLAUNCH_PID" 2>/dev/null; then
      return 1
    fi
    sleep 0.1
  done
  return 1
}

require_file /opt/ros/noetic/setup.bash
require_file "$REPOSITORY_ROOT/devel/setup.bash"
require_file "$BAG_PATH"

# ROS Noetic must use the system Python environment, not the active Conda Python.
source /opt/ros/noetic/setup.bash
source "$REPOSITORY_ROOT/devel/setup.bash"

for command_name in roscore rosnode rosparam roslaunch rosbag; do
  require_command "$command_name"
done

if ! rosnode list >/dev/null 2>&1; then
  echo "[bev] ROS master가 없어 이 실행 전용 roscore를 시작합니다."
  roscore >"$ROSCORE_LOG" 2>&1 &
  ROSCORE_PID=$!
  if ! wait_for_master; then
    tail -n 30 "$ROSCORE_LOG" >&2 || true
    fail "ROS master 시작에 실패했습니다."
  fi
else
  echo "[bev] 현재 실행 중인 ROS master를 재사용합니다."
fi

if rosnode list 2>/dev/null | grep -Fxq "/lidar_attitude_debugger"; then
  fail "/lidar_attitude_debugger가 이미 실행 중입니다. 기존 실행을 먼저 종료하세요."
fi

rosparam set /use_sim_time true

echo "[bev] LiDAR 자세 디버거와 RViz를 시작합니다."
roslaunch Validator_pkg lidar_attitude_debug.launch show_rviz:=true \
  >"$ROSLAUNCH_LOG" 2>&1 &
ROSLAUNCH_PID=$!

if ! wait_for_debugger; then
  tail -n 50 "$ROSLAUNCH_LOG" >&2 || true
  fail "LiDAR 자세 디버거 시작에 실패했습니다."
fi

echo "[bev] rosbag 반복 재생을 시작합니다: $BAG_PATH"
echo "[bev] 종료하려면 이 터미널에서 Ctrl+C를 누르세요."
rosbag play --clock --loop "$BAG_PATH"
