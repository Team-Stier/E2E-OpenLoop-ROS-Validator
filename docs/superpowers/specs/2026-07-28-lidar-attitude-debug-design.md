# LiDAR Roll/Pitch 진단 시각화 설계

## 1. 목적

MORAI rosbag의 `/lidar3D`와 `/tf`를 사용해 다음 두 자세를 RViz에서 함께 관찰한다.

- `map` 기준 차량 자세: `/tf`의 `map -> base_link` roll/pitch
- 지역 도로면 기준 차량 자세: LiDAR 지면 평면으로 추정한 roll/pitch

이 기능의 목적은 카메라/LiDAR BEV 흔들림과 roll/pitch 관계를 진단하는 것이다. 제어기나 위치추정기에 공급할 절대 자세 추정기를 만드는 것은 범위에 포함하지 않는다.

현재 rosbag에는 IMU 토픽이 없으므로 IMU 융합은 수행하지 않는다. 경사로나 편경사 구간에서는 TF 자세와 LiDAR 도로면 상대 자세의 차이에 실제 도로 경사도 포함된다는 점을 화면에 명시한다.

## 2. 현재 계약과 제약

- 원본 LiDAR 토픽은 `/lidar3D`, 타입은 `sensor_msgs/PointCloud2`, frame은 `Lidar3D-4`다.
- `src/Validator_pkg/validator_pipeline/config/0716demo.json`의 LiDAR 고정 장착값은 `base_link` 기준 위치 `(2.0, 0.0, 1.5) m`, RPY `(0, 0, 0) deg`다.
- 기존 `/lidar3D/BEV`는 `mono8`, 600 x 500 시각화 영상이다.
- 기존 모델별 LiDAR 입력과 `/lidar3D/BEV`의 이름, 타입, 값 범위는 변경하지 않는다.
- 기존 `lidar_bev_processor.py`에는 진단 기능을 추가하지 않는다. 새 진단 노드는 필요할 때만 별도 실행한다.
- 진단 파라미터는 코드에 흩어 놓지 않고 전용 YAML에서 관리한다.

## 3. 선택한 구조

새 ROS 노드 `lidar_attitude_debugger.py`를 `Validator_pkg`에 추가한다.

```text
/lidar3D ──────┐
               ├─ lidar_attitude_debugger
/tf ───────────┘
        ├─ /lidar/debug/ground_bev
        ├─ /lidar/debug/colored_points
        └─ /lidar/debug/markers
```

별도 노드를 선택한 이유는 전처리, 모델 입력, 진단 책임을 분리하고 대회 실행 경로에 영향을 주지 않기 위해서다.

## 4. 입력

### 4.1 `/lidar3D`

- 타입: `sensor_msgs/PointCloud2`
- 필수 필드: `x`, `y`, `z`
- 원본 frame: `Lidar3D-4`
- 각 포인트에 `0716demo.json`의 고정 장착 회전과 평행이동을 적용해 `base_link` 좌표로 변환한다.
- NaN, Inf 및 `(0, 0, 0)` 반환은 제거한다.

### 4.2 `/tf`

- 타입: `tf2_msgs/TFMessage`
- 사용할 변환: `map -> base_link`
- LiDAR stamp 이하의 최신 TF만 사용한다.
- TF age가 설정된 최대값을 초과하면 TF 자세를 `INVALID`로 표시한다.
- rosbag 반복 재생으로 stamp가 뒤로 이동하면 TF와 평활화 이력을 초기화한다.

## 5. 지면과 자세 추정

### 5.1 빨간색 z=0 띠

`base_link`로 변환한 포인트 중 다음 조건을 만족하는 점을 빨간색으로 분류한다.

```text
abs(z) <= 0.10 m
```

임계값 `0.10 m`는 YAML 파라미터 `ground_band_half_width`로 변경할 수 있다. 정확히 `z == 0`인 점만 선택하지 않는 이유는 PointCloud2 측정과 양자화에서 정확한 0이 거의 나타나지 않기 때문이다.

### 5.2 지면 후보

RANSAC 입력은 다음 초기 ROI로 제한한다.

```text
x: 2.0 m 이상 25.0 m 이하
y: -10.0 m 이상 10.0 m 이하
z: -0.75 m 이상 0.75 m 이하
```

모든 범위는 YAML에서 조정할 수 있다. 차량 자체와 먼 거리의 희박한 점, 높은 장애물이 평면 추정을 지배하지 않도록 ROI를 적용한다.

### 5.3 RANSAC 평면

지면 후보에 대해 다음 평면을 추정한다.

```text
a*x + b*y + c*z + d = 0
```

법선은 위쪽을 향하도록 `n_z >= 0`으로 정규화한다. 초기 유효성 기준은 다음과 같다.

- RANSAC 거리 임계값: `0.15 m`
- 최소 후보 포인트: 100개
- 최소 inlier 비율: 0.20
- 허용 최대 지면 기울기: 15 deg
- 허용 최대 inlier RMS: `0.15 m`

기준을 하나라도 충족하지 못하면 LiDAR 자세를 `INVALID`로 처리하며 이전의 유효 각도를 현재 값처럼 재사용하지 않는다.

### 5.4 LiDAR 도로면 상대 자세

정규화된 지면 법선 `n = (n_x, n_y, n_z)`에서 다음과 같이 계산한다.

```text
roll_lidar  = atan2(n_y, n_z)
pitch_lidar = atan2(-n_x, sqrt(n_y^2 + n_z^2))
```

유효 결과에는 5프레임 median filter를 적용한다. 필터 적용 전 raw 값과 적용 후 값을 혼동하지 않도록 RViz에는 filtered 값을 표시하고 로그에는 raw/filtered 값을 구분한다.

### 5.5 TF world 자세

`map -> base_link` quaternion을 ROS RPY 관례로 변환한다. 이 값은 world-relative 자세로 표시하며 LiDAR road-relative 자세와 동일한 물리량이라고 표현하지 않는다.

## 6. 출력

### 6.1 `/lidar/debug/ground_bev`

- 타입: `sensor_msgs/Image`
- encoding: `bgr8`
- 크기: 기존 BEV와 동일한 600 x 500
- 좌표 범위: x `[0, 25] m`, y `[-15, 15] m`
- frame: `base_link`
- 일반 LiDAR 점: 회색
- `abs(z) <= ground_band_half_width` 점: 빨간색
- 좌측 상단 텍스트:
  - `TF world roll/pitch`
  - `LiDAR road-relative roll/pitch`
  - TF age
  - RANSAC 후보 수와 inlier 비율
  - plane RMS
  - 각 추정의 `VALID` 또는 `INVALID`

### 6.2 `/lidar/debug/colored_points`

- 타입: `sensor_msgs/PointCloud2`
- 필드: `x`, `y`, `z`, `rgb`
- frame: `base_link`
- 일반 점: 회색
- `abs(z) <= ground_band_half_width` 점: 빨간색

RViz의 PointCloud2 display는 RGB transformer를 사용한다.

### 6.3 `/lidar/debug/markers`

- 타입: `visualization_msgs/MarkerArray`
- frame: `base_link`
- 반투명 기준 `z=0` 평면
- LiDAR 추정 지면 법선: 노란색 화살표
- TF world-up을 `base_link`로 표현한 벡터: 청록색 화살표
- TF/LiDAR 상태와 수치를 표시하는 text marker

추정이 유효하지 않으면 해당 법선 화살표를 삭제하고 text marker에 원인을 표시한다.

## 7. 설정과 실행

다음 파일을 추가한다.

- `config/lidar_attitude_debug.yaml`
- `launch/lidar_attitude_debug.launch`
- `rviz/lidar_attitude_debug.rviz`

실행 명령은 다음과 같다.

```bash
roslaunch Validator_pkg lidar_attitude_debug.launch
```

launch는 진단 노드와 전용 RViz 설정을 시작한다. rosbag과 `/use_sim_time`은 기존 실행 방식대로 외부에서 공급한다. 기존 `camera_bev_pipeline.launch`, `bev_visualization.launch`, `openloop_validator.launch`는 변경하지 않는다.

## 8. 오류 처리

- PointCloud2에 `x/y/z` 필드가 없으면 throttled error를 남기고 해당 프레임을 건너뛴다.
- 빈 cloud 또는 유효점이 없으면 빈 진단 영상을 발행하지 않고 상태를 `INVALID`로 표시한다.
- TF가 없거나 오래되면 LiDAR 기반 출력은 계속 발행하고 TF 항목만 `INVALID`로 표시한다.
- 지면 후보가 부족하거나 RANSAC이 실패하면 빨간 z=0 시각화는 계속 발행하고 LiDAR 자세와 법선만 `INVALID`로 표시한다.
- stamp가 뒤로 이동하면 TF deque와 median filter를 초기화해 이전 반복의 자세가 섞이지 않게 한다.
- 출력 message는 원본 LiDAR header stamp를 유지한다.

## 9. 의존성과 인터페이스 영향

- `Validator_pkg/package.xml`과 `CMakeLists.txt`에 `visualization_msgs` 런타임 의존성과 새 Python 실행 파일 설치를 추가한다.
- 새 토픽만 추가하며 기존 토픽의 타입, frame, 주기 및 소비자를 변경하지 않는다.
- 원본 센서 JSON과 rosbag은 수정하지 않는다.
- 진단 노드는 Controller, E2E Planner, Localization 및 모델 inference에 연결하지 않는다.

## 10. 검증

### 10.1 단위 테스트

- 수평 합성 평면에서 roll/pitch가 0에 가까운지 확인
- 알려진 roll/pitch 합성 평면에서 법선 기반 각도 복원 확인
- 장애물 outlier가 섞여도 RANSAC이 지면을 선택하는지 확인
- 최소 포인트, 낮은 inlier 비율, 큰 RMS가 `INVALID`가 되는지 확인
- `z` 경계 안의 점만 빨간색으로 분류되는지 확인
- quaternion에서 TF roll/pitch 변환 확인
- TF 누락, stale TF, 시간 역행 시 상태와 이력 초기화 확인

### 10.2 빌드와 정적 계약

- `/usr/bin/python3`를 사용한 catkin 빌드
- Python syntax 검사
- package manifest와 install contract 검사
- 기존 preprocessing/inference 테스트 회귀 확인

### 10.3 rosbag 실행 검증

- 현재 `2026-07-16-17-13-14.bag`을 `--clock --loop`로 재생
- 세 진단 토픽의 실제 발행 확인
- RViz에서 빨간 z=0 띠, 기준 평면, 두 법선 및 상태 텍스트 확인
- bag의 `map -> base_link` roll/pitch 범위와 표시값이 일치하는지 확인
- 평지와 경사 구간에서 TF world 자세와 LiDAR road-relative 자세의 차이를 구분해 관찰

IMU 또는 외부 자세 기준값이 없으므로 절대 각도 정확도는 검증 완료로 표현하지 않는다. 이번 검증의 완료 기준은 계산·상태 처리·토픽 계약과 시각적 일관성이다.
