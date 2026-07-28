# `bev` Bash 명령 설계

## 목적

새 터미널에서 `bev` 한 단어를 입력하면 LiDAR 자세 진단 RViz와
`2026-07-16-17-13-14.bag` 반복 재생을 함께 시작한다.

## 구조

- `~/.bashrc`의 `bev()` 함수는 저장소의 실행 스크립트만 호출한다.
- 실제 실행과 정리는 `scripts/run_lidar_bev_debug.sh`가 담당한다.
- 기존 ROS master가 있으면 재사용하고, 없으면 스크립트가 시작한다.
- 스크립트가 시작한 프로세스만 `Ctrl+C`와 종료 trap에서 정리한다.

## 실행 전 검사

- `/opt/ros/noetic/setup.bash`
- 저장소 `devel/setup.bash`
- 지정 rosbag 파일
- `roscore`, `rosnode`, `rosparam`, `roslaunch`, `rosbag`
- 기존 `/lidar_attitude_debugger` 노드 충돌

검사가 실패하면 RViz나 rosbag을 일부만 실행하지 않고 원인을 출력한다.

## 실행 순서

1. ROS Noetic과 workspace overlay를 source한다.
2. ROS master를 확인하고 필요할 때만 시작한다.
3. `/use_sim_time=true`를 설정한다.
4. `lidar_attitude_debug.launch`를 RViz와 함께 백그라운드 실행한다.
5. 진단 노드가 나타날 때까지 제한시간 동안 확인한다.
6. rosbag을 `--clock --loop`로 foreground 재생한다.
7. 종료 시 이번 명령이 시작한 roslaunch와 roscore만 정리한다.

## `.bashrc` 변경

기존 내용을 보존하고 다음 marker 블록만 한 번 추가한다.

```bash
# >>> Team-Stier LiDAR BEV debug >>>
bev() {
    /home/stier/rosbag_lidar_horizental/E2E-OpenLoop-ROS-Validator/scripts/run_lidar_bev_debug.sh
}
# <<< Team-Stier LiDAR BEV debug <<<
```

현재 셸에서는 `source ~/.bashrc` 이후 사용할 수 있고, 새 터미널에서는
자동으로 사용할 수 있다.
