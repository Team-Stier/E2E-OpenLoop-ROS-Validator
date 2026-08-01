# horizontal_pkg

ROS1 Noetic rosbag에서 IMU 자세를 분석·필터링하고, LiDAR 포인트를 roll/pitch 기준으로 완전히 수평화하는 오프라인 패키지다. 원본 LiDAR와 보정 LiDAR를 겹쳐 보는 비교 MP4도 생성한다.

## 주요 역할

- `/imu` quaternion에서 roll/pitch 추출
- 4 Hz 2차 Butterworth 저역통과 필터 적용
- bag의 `/tf`에서 IMU–LiDAR 고정 기하 추출
- `/lidar3D`의 XYZ를 필터링된 roll/pitch로 회전
- 출력 bag에 `/lidar3D/horizontal` 추가
- 원본·보정 LiDAR의 X–Z/Y–Z 비교 애니메이션 생성

거리 측정값 자체의 노이즈를 제거하는 패키지가 아니라, 차량 자세 변화로 기울어진 LiDAR 좌표를 보정하는 패키지다.

## ROS 토픽

| 토픽 | 메시지 타입 | 방향 | 역할 |
|---|---|---|---|
| `/imu` | `sensor_msgs/Imu` | 입력 | quaternion과 timestamp 제공 |
| `/tf` | `tf2_msgs/TFMessage` | 입력 | `base_link` 기준 IMU·LiDAR 고정 위치/회전 제공 |
| `/lidar3D` | `sensor_msgs/PointCloud2` | 입력·원본 유지 | 보정 전 LiDAR 포인트 |
| `/lidar3D/horizontal` | `sensor_msgs/PointCloud2` | 출력 bag에 추가 | roll/pitch 수평 보정 LiDAR 포인트 |

이 패키지는 현재 실시간 publisher가 아니다. 생성한 출력 bag을 `rosbag play`할 때 `/lidar3D/horizontal`이 ROS 토픽으로 발행된다.

## 좌표계

| frame | 역할 |
|---|---|
| `base_link` | 차량 기준 좌표계 및 센서 TF parent |
| `IMU-1` | 입력 IMU frame |
| `Lidar3D-4` | 원본 LiDAR frame |
| `Lidar3D-4-horizontal` | 완전 수평화된 출력 LiDAR frame |

현재 rosbag에서 확인한 센서 위치는 다음과 같다.

- `base_link → IMU-1`: `[1.5, 0.0, 0.5] m`
- `base_link → Lidar3D-4`: `[1.5, 0.0, 1.25] m`
- LiDAR는 IMU보다 약 `0.75 m` 위에 있다.

## 처리 흐름

```text
/imu ── quaternion → roll/pitch → 4 Hz filtering ─┐
/tf  ── IMU–LiDAR 고정 기하 ──────────────────────┤
/lidar3D ──────────────────────────────────────────┼→ XYZ 회전
                                                   └→ /lidar3D/horizontal
```

원본 bag의 메시지는 유지하면서 같은 record time에 보정 토픽을 추가한다. 기본 IMU 정렬 허용 간격은 `0.10 s`이며, 안전하게 정렬할 수 없는 LiDAR 프레임은 건너뛰고 개수를 보고한다.

## LiDAR 수평화 bag 생성

catkin workspace 루트에서 실행한다.

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash

/usr/bin/python3 -m horizontal_pkg.lidar_horizontal \
  --input-bag /path/to/input.bag \
  --output-bag /path/to/output_horizontal.bag
```

주요 기본값:

- IMU 필터 cutoff: `4.0 Hz`
- IMU 최대 정렬 gap: `0.10 s`
- 원본 토픽: `/lidar3D`
- 출력 토픽: `/lidar3D/horizontal`
- 출력 frame: `Lidar3D-4-horizontal`

출력 확인:

```bash
rosbag info /path/to/output_horizontal.bag
rosbag play /path/to/output_horizontal.bag
rostopic hz /lidar3D/horizontal
```

## 비교 애니메이션 생성

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash

/usr/bin/python3 -m horizontal_pkg.lidar_comparison_animation \
  --input-bag /path/to/output_horizontal.bag \
  --output-mp4 src/Validator_pkg/horizontal_pkg/docs/lidar_horizontal_comparison.mp4
```

기본 출력은 전체 주행에서 600프레임을 균등 선택해 10 FPS로 만든다. 한 화면에 최대 12,000개의 동일 point index를 표시한다.

- 원본 `/lidar3D`: 주황-빨강
- 보정 `/lidar3D/horizontal`: 청록
- 좌측 X–Z: pitch 보정 비교
- 우측 Y–Z: roll 보정 비교

## 결과 자료

- [IMU roll/pitch](docs/imu_roll_pitch.png)
- [필터 전·후 IMU roll/pitch](docs/imu_roll_pitch_filtered.png)
- [원본·보정 LiDAR 비교 애니메이션](docs/lidar_horizontal_comparison.mp4)

## 주의사항

- 오프라인 rosbag 처리 전용이며 MORAI에서 실시간으로 publish하는 노드가 아니다.
- XYZ를 강체 회전하므로 LiDAR 거리 노이즈나 평면 잔차 자체는 감소하지 않는다.
- 완전 수평화 방식이므로 차량 흔들림뿐 아니라 실제 오르막·내리막 경사도 제거한다.
- 입력 IMU/LiDAR frame, TF, timestamp 또는 PointCloud2 layout이 계약과 다르면 실패한다.
- 원본 `/lidar3D`는 수정하지 않고 별도 `/lidar3D/horizontal` 토픽을 만든다.
