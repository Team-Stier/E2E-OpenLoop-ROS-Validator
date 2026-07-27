# BEV Visualization Setup

이 문서는 `E2E-OpenLoop-ROS-Validator` 프로젝트에서 Camera BEV, LiDAR BEV, inference waypoint를 하나의 화면으로 합성하고 RViz에서 확인하기 위해 추가된 파일 구조, ROS 의존성, 빌드 및 실행 방법을 설명한다.

---
현재 파일의 위치에 맞게 수정해두긴 했다. 혹시나 이상이 있으면 김지한에게 알려주세요
---

## 1. 목적

시각화 계층은 다음 세 데이터를 사용한다.

| 입력 | ROS 토픽 | 메시지 타입 |
|---|---|---|
| Camera BEV | `/image/BEV` | `sensor_msgs/Image`, `bgr8` |
| LiDAR BEV | `/lidar3D/BEV` | `sensor_msgs/Image`, `mono8` |
| Model waypoint | `/waypoints` | `nav_msgs/Path`, `Lidar3D-4` |

Visualizer 노드는 위 입력을 구독해 다음 결과를 발행한다.

| 출력 | ROS 토픽 | 메시지 타입 |
|---|---|---|
| BEV 합성 화면 | `/visualization/bev_overlay` | `sensor_msgs/Image`, `bgr8` |
| 입력·합성 비교 화면 | `/visualization/bev_triptych` | `sensor_msgs/Image`, `bgr8` |

`/visualization/bev_overlay`에는 Camera BEV를 배경으로 LiDAR BEV와 waypoint가 중첩된다.

- Camera BEV: 원래 RGB 색상
- LiDAR BEV: 초록색
- Waypoint 연결선: 파란색
- Waypoint 위치점: 노란색

`/visualization/bev_triptych`에는 아래 세 영상이 좌우로 배치된다.

```text
┌────────────────┬────────────────┬────────────────┐
│ Camera BEV     │ LiDAR BEV      │ Overlay        │
└────────────────┴────────────────┴────────────────┘
```

이 화면은 모델의 객체 인식 결과가 아니라, Camera/LiDAR BEV 정합과 inference waypoint 위치를 검증하기 위한 디버깅 화면이다.

---

## 2. 추가 파일 구조

`src/Validator_pkg` 자체는 manifest가 없는 Catkin package group이다. `validator_inference`, `validator_msgs`, `validator_pipeline` 세 package가 sibling으로 있고, preprocessing/visualization 구현은 `validator_pipeline`에 둔다. 이 디렉터리의 `package.xml` 이름은 launch 호환을 위해 `Validator_pkg`다.

```text
E2E-OpenLoop-ROS-Validator/
└── src/
    └── Validator_pkg/
        ├── config/profile.json
        ├── validator_inference/
        ├── validator_msgs/
        └── validator_pipeline/        # ROS package: Validator_pkg
            ├── scripts/
            │   └── bev_visualizer.py
            ├── config/
            │   ├── bev_geometry.yaml
            │   └── bev_visualizer.yaml
            ├── launch/
            │   ├── bev_visualization.launch
            │   └── openloop_validator.launch
            ├── rviz/
            │   └── e2e_validator.rviz
            ├── CMakeLists.txt
            └── package.xml
```

각 파일의 역할은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `scripts/bev_visualizer.py` | Camera BEV와 LiDAR BEV를 동기화하고 waypoint를 포함해 합성 |
| `config/bev_geometry.yaml` | 공통 BEV 좌표계, 범위, 해상도 및 이미지 크기 정의 |
| `config/bev_visualizer.yaml` | 구독·발행 토픽, 합성 비율, 동기화 허용시간 등 정의 |
| `launch/bev_visualization.launch` | Visualizer와 RViz 실행 |
| `launch/openloop_validator.launch` | Preprocessing, inference, visualization 통합 실행 |
| `rviz/e2e_validator.rviz` | RViz Display 구성 저장 |
| `CMakeLists.txt` | Python 노드 및 설정 폴더 설치, ROS 패키지 의존성 선언 |
| `package.xml` | 빌드·실행 시 필요한 ROS 패키지 의존성 선언 |

---

## 3. 공통 BEV 좌표 계약

Camera BEV와 LiDAR BEV는 단순히 영상 크기만 같아서는 안 된다. 다음 조건을 모두 동일하게 사용해야 한다.

```text
frame_id        : base_link
x축             : 차량 전방
y축             : 차량 좌측
단위            : m
metric bounds   : x=[0,25] m, y=[-15,15] m
resolution      : 0.05 m/pixel
image size      : width=600, height=500
image 위쪽      : 차량 전방
image 왼쪽      : 차량 좌측
timestamp       : 원본 센서 timestamp 유지
```

이 설정은 `config/bev_geometry.yaml`에 정의하며, 다음 노드가 공통으로 읽어야 한다.

```text
camera_bev_stitcher
lidar_bev_processor
bev_visualizer
```

Visualizer는 서로 다른 Camera/LiDAR BEV 기하를 resize로 맞추는 노드가 아니다. 단, inference의 `/waypoints`는 계약상 `Lidar3D-4` frame으로 발행되므로, `bev_geometry.yaml`에 기록된 `T_base_from_lidar`(평행이동 `[2,0,1.5]`, quaternion `[0,0,0,1]`)를 적용해 waypoint만 `base_link` BEV로 변환한다.

```text
Preprocessing layer:
    Camera BEV와 LiDAR BEV를 동일한 기하학 조건으로 생성

Visualization layer:
    동일한 기하학 조건으로 생성된 두 BEV를 시간 동기화 후 합성
```

따라서 Camera BEV와 LiDAR BEV의 크기가 다를 때 Visualizer에서 `cv2.resize()`로 강제로 맞추면 안 된다. 각 processor의 BEV 범위와 해상도 설정을 수정해야 한다.

---

## 4. `package.xml` 의존성

BEV Visualizer는 다음 ROS 패키지를 사용한다.

| 패키지 | 용도 |
|---|---|
| `rospy` | Python ROS 노드 |
| `sensor_msgs` | Camera·LiDAR BEV Image |
| `nav_msgs` | Model waypoint `Path` |
| `cv_bridge` | ROS Image와 OpenCV 영상 변환 |
| `message_filters` | Camera BEV와 LiDAR BEV 시간 동기화 |
| `rviz` | 합성 결과 시각화 |

`package.xml`에는 최소한 다음 항목이 필요하다.

```xml
<build_depend>cv_bridge</build_depend>
<build_depend>message_filters</build_depend>
<build_depend>nav_msgs</build_depend>
<build_depend>rospy</build_depend>
<build_depend>sensor_msgs</build_depend>

<build_export_depend>cv_bridge</build_export_depend>
<build_export_depend>message_filters</build_export_depend>
<build_export_depend>nav_msgs</build_export_depend>
<build_export_depend>rospy</build_export_depend>
<build_export_depend>sensor_msgs</build_export_depend>

<exec_depend>cv_bridge</exec_depend>
<exec_depend>message_filters</exec_depend>
<exec_depend>nav_msgs</exec_depend>
<exec_depend>rospy</exec_depend>
<exec_depend>sensor_msgs</exec_depend>
<exec_depend>rviz</exec_depend>
```

기존 프로젝트에서 이미 선언된 `morai_msgs`, `std_msgs`, `tf2_msgs` 의존성은 그대로 유지한다.

---

## 5. `CMakeLists.txt` 설정

### 5.1 `find_package`

`nav_msgs`를 포함해야 한다.

```cmake
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  message_filters
  morai_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_msgs
)
```

### 5.2 `catkin_package`

```cmake
catkin_package(
  CATKIN_DEPENDS
    cv_bridge
    message_filters
    morai_msgs
    nav_msgs
    rospy
    sensor_msgs
    std_msgs
    tf2_msgs
)
```

### 5.3 Python 노드 설치

`bev_visualizer.py`를 `rosrun`과 `roslaunch`에서 실행할 수 있도록 등록한다.

```cmake
catkin_install_python(PROGRAMS
  scripts/image_adaptor.py
  scripts/camera_bev_stitcher.py
  scripts/front_bev_processor.py
  scripts/left_bev_processor.py
  scripts/right_bev_processor.py
  scripts/lidar_bev_processor.py
  scripts/bev_overlay_viewer.py
  scripts/bev_visualizer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
```

프로젝트에서 `bev_overlay_viewer.py`가 이미 동일한 역할을 수행한다면 두 파일을 동시에 유지할 필요는 없다. 실제로 사용하는 파일명 하나만 CMake에 등록한다.

### 5.4 설정 및 RViz 파일 설치

```cmake
install(DIRECTORY config launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
```

이 설정을 사용하려면 `config/`, `launch/`, `rviz/` 폴더가 실제로 존재해야 한다.

---

## 6. 실행 권한

Python 노드는 실행 권한이 필요하다.

```bash
chmod +x \
~/catkin_ws/src/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/validator_pipeline/scripts/bev_visualizer.py
```

실행 권한을 확인한다.

```bash
ls -l \
~/catkin_ws/src/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/validator_pipeline/scripts/bev_visualizer.py
```

파일 권한에 `x`가 포함되어 있어야 한다.

```text
-rwxr-xr-x
```

---

## 7. 빌드

패키지가 다음 경로에 있다고 가정한다.

```text
~/catkin_ws/src/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/validator_pipeline
```

먼저 Python 문법을 검사한다.

```bash
python3 -m py_compile \
~/catkin_ws/src/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/validator_pipeline/scripts/bev_visualizer.py
```

워크스페이스를 빌드한다.

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

새 터미널을 열 때마다 다음 명령을 다시 실행해야 한다.

```bash
source ~/catkin_ws/devel/setup.bash
```

매번 수동으로 입력하지 않으려면 `~/.bashrc`에 추가할 수 있다.

```bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

의존성 누락 여부는 다음 명령으로 확인한다.

```bash
cd ~/catkin_ws

rosdep install \
  --from-paths src \
  --ignore-src \
  -r \
  -y
```

---

## 8. RViz 설정

Visualizer가 발행하는 합성 영상은 RViz의 3D 바닥면에 직접 표시되는 것이 아니라 `Image` Display 패널에 표시된다.

RViz를 직접 실행하는 경우:

```bash
rviz
```

### 8.1 합성 화면

RViz에서 다음 순서로 추가한다.

```text
Add
→ Image
→ Image Topic: /visualization/bev_overlay
```

표시 내용:

```text
Camera BEV
+ 초록색 LiDAR BEV
+ 파란색 waypoint 선
+ 노란색 waypoint 점
```

### 8.2 입력·결과 비교 화면

Image Display를 하나 더 추가한다.

```text
Add
→ Image
→ Image Topic: /visualization/bev_triptych
```

### 8.3 3D 데이터 추가

다음 Display를 추가하면 원본 LiDAR와 model waypoint를 3D View에서 함께 볼 수 있다.

| RViz Display | Topic |
|---|---|
| `PointCloud2` | `/lidar3D` |
| `Path` | `/waypoints` |
| `TF` | `/tf` |
| `Image` | `/visualization/bev_overlay` |
| `Image` | `/visualization/bev_triptych` |

RViz의 `Fixed Frame`은 다음으로 설정한다.

```text
base_link
```

Inference layer의 `/waypoints`는 `Lidar3D-4` frame으로 발행된다. BEV Visualizer는 공유 설정의 정적 `T_base_from_lidar`를 적용해 `base_link`로 변환한 뒤 합성한다. RViz 3D Path Display도 동일한 frame tree를 볼 수 있어야 한다.

---

## 9. 실행 순서

### 통합 실행(권장)

저장소 root에서 모델 환경을 한 번 준비한 뒤 validator와 RViz를 시작한다.

```bash
./openloop setup interfuser
./openloop interfuser --rviz
```

별도 terminal에서 기본 bag과 `/clock`을 재생한다.

```bash
./openloop bag
```

다른 모델은 해당 모델 환경 준비가 완료된 경우에만 `interfuser`를 해당 model ID로 바꾼다. SparseDrive V2의 front/left/right inline calibration은 shipped profile에 포함되어 있다. Bag에는 `CameraInfo`가 없지만, 저장된 MORAI sensor JSON의 image size/FOV/principal point와 bag `/tf`를 함께 사용해 값을 확정했으므로 별도로 채울 항목은 없다.

### 고급 직접 launch

CLI가 전달하는 경로를 직접 제어할 때만 ROS setup을 source한 terminal에서 다음 형식을 사용한다. 먼저 해당 모델의 `./openloop setup MODEL_ID`를 완료해야 한다.

```bash
roslaunch Validator_pkg openloop_validator.launch \
  project_root:=/home/stier/E2E-OpenLoop-ROS-Validator \
  profile_file:=/home/stier/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/config/profile.json \
  sensor_config:=/home/stier/E2E-OpenLoop-ROS-Validator/src/Validator_pkg/validator_pipeline/config/0716demo.json \
  model_id:=interfuser \
  model_env_root:=/home/stier/E2E-OpenLoop-ROS-Validator/.venvs \
  show_rviz:=true
```

아래 절도 정상 운영 절차가 아니라 각 component를 분리해 점검하는 고급 진단 절차다.

### 터미널 1: ROS Master

```bash
roscore
```

### 터미널 2: Preprocessing layer

rosbag timestamp를 사용하도록 시뮬레이션 시간을 활성화한다.

```bash
rosparam set use_sim_time true
```

TFv6 profile로 preprocessing pipeline을 실행한다.

```bash
roslaunch Validator_pkg camera_bev_pipeline.launch \
  model_profiles:=tfv6 \
  show_gui:=false
```

이 단계에서 최소한 다음 토픽이 발행되어야 한다.

```text
/image/BEV
/lidar3D/BEV
```

### 터미널 3: Visualizer와 RViz

Inference 노드가 실행 중이고 `/waypoints`가 발행되는 경우:

```bash
roslaunch Validator_pkg bev_visualization.launch \
  show_rviz:=true \
  draw_waypoints:=true
```

아직 inference 노드를 연결하지 않은 경우:

```bash
roslaunch Validator_pkg bev_visualization.launch \
  show_rviz:=true \
  draw_waypoints:=false
```

### 터미널 4: rosbag 재생

```bash
cd ~/catkin_ws/src/E2E-OpenLoop-ROS-Validator

rosbag play \
  rosbag/2026-07-16-17-13-14.bag \
  --clock
```

재생 속도를 낮춰 확인하려면 다음과 같이 실행한다.

```bash
rosbag play \
  rosbag/2026-07-16-17-13-14.bag \
  --clock \
  -r 0.5
```

---

## 10. 토픽 연결 확인

### 10.1 입력 토픽 주기

```bash
rostopic hz /image/BEV
rostopic hz /lidar3D/BEV
```

### 10.2 합성 결과 주기

```bash
rostopic hz /visualization/bev_overlay
rostopic hz /visualization/bev_triptych
```

### 10.3 인코딩과 영상 크기

```bash
rostopic echo -n 1 /image/BEV/encoding
rostopic echo -n 1 /image/BEV/width
rostopic echo -n 1 /image/BEV/height

rostopic echo -n 1 /lidar3D/BEV/encoding
rostopic echo -n 1 /lidar3D/BEV/width
rostopic echo -n 1 /lidar3D/BEV/height
```

현재 계약 기준 기대값은 다음과 같다.

```text
/image/BEV
encoding: bgr8
width: 600
height: 500

/lidar3D/BEV
encoding: mono8
width: 600
height: 500
```

### 10.4 Timestamp

```bash
rostopic echo -n 1 /image/BEV/header
rostopic echo -n 1 /lidar3D/BEV/header
```

두 토픽의 timestamp가 `bev_visualizer.yaml`에 설정한 `sync_slop` 범위 안에 있어야 한다.

---

## 11. 문제 진단

### `/visualization/bev_overlay`가 발행되지 않는 경우

다음 항목을 순서대로 확인한다.

```bash
rostopic list | grep BEV
rostopic hz /image/BEV
rostopic hz /lidar3D/BEV
rosnode info /bev_visualizer
```

주요 원인:

- `/image/BEV` 또는 `/lidar3D/BEV`가 발행되지 않음
- 두 토픽의 timestamp 차이가 `sync_slop`보다 큼
- 두 BEV의 크기가 다름
- Camera BEV 인코딩이 `bgr8`이 아님
- LiDAR BEV 인코딩이 `mono8`이 아님
- `bev_visualizer.py` 실행 권한 없음

### 두 BEV가 일정 거리만큼 어긋나는 경우

가능성이 높은 원인:

```text
Camera 또는 LiDAR extrinsic translation
서로 다른 BEV 원점
서로 다른 x_min, x_max, y_min, y_max
```

### 두 BEV가 회전되어 어긋나는 경우

가능성이 높은 원인:

```text
센서 yaw 설정
x·y축 convention 차이
이미지 row·column 방향 차이
```

### 정지 상태에서는 맞고 주행 중 어긋나는 경우

가능성이 높은 원인:

```text
Camera와 LiDAR timestamp 차이
rosbag --clock 누락
processor에서 rospy.Time.now() 사용
과거 LiDAR sweep motion compensation 오류
```

### Waypoint가 표시되지 않는 경우

확인 사항:

```bash
rostopic hz /waypoints
rostopic echo -n 1 /waypoints/header
```

정상 inference 출력의 `/waypoints.header.frame_id`는 `Lidar3D-4`다. visualizer는 inference와 동일한 `profile.json`의 `common_inference_policy.static_transforms.T_base_from_lidar`를 읽는다. parent/child가 각각 `base_link`/`Lidar3D-4`인지, translation이 `[2.0, 0.0, 1.5]`, quaternion이 `[0.0, 0.0, 0.0, 1.0]`인지 확인한다. 다른 frame이 들어오면 묵시한 채 그리지 말고 해당 frame에 맞는 명시적 변환을 profile에 기록해야 한다.

---

## 12. 각 파일의 책임

| 조건 | 담당 파일 또는 노드 |
|---|---|
| Camera BEV 생성 | `camera_bev_stitcher.py` |
| LiDAR BEV 생성 | `lidar_bev_processor.py` |
| 동일한 미터 범위와 해상도 | `bev_geometry.yaml` |
| Camera·LiDAR 시간 동기화 | `bev_visualizer.py` |
| 두 BEV 색상 합성 | `bev_visualizer.py` |
| Waypoint 픽셀 변환 및 표시 | `bev_visualizer.py` |
| 토픽·색상·동기화 파라미터 | `bev_visualizer.yaml` |
| Visualizer 실행 | `bev_visualization.launch` |
| RViz Display 저장 | `e2e_validator.rviz` |
| Python 노드 및 폴더 설치 | `CMakeLists.txt` |
| ROS 의존성 선언 | `package.xml` |

---

## 13. 최종 데이터 흐름

```text
rosbag
│
├─ /image_jpeg/front/compressed
├─ /image_jpeg/left/compressed
├─ /image_jpeg/right/compressed
└─ /lidar3D
        │
        ▼
Preprocessing layer
│
├─ camera_bev_stitcher
│      └─ /image/BEV
│
└─ lidar_bev_processor
       └─ /lidar3D/BEV
        │
        ▼
bev_visualizer
│
├─ /visualization/bev_overlay
└─ /visualization/bev_triptych
        │
        ▼
RViz Image Display
```

Inference engine이 실행 중이면 다음 경로가 추가된다.

```text
inference engine
└─ /waypoints (Lidar3D-4)
        |
        | T_base_from_lidar
        v
bev_visualizer (base_link BEV)
└─ Camera + LiDAR + Waypoint 합성
```

---

## 14. 검증 기준

정상적으로 구성된 경우 다음 조건이 모두 만족되어야 한다.

```text
[ ] /image/BEV가 bgr8, 600 x 500으로 발행됨
[ ] /lidar3D/BEV가 mono8, 600 x 500으로 발행됨
[ ] 두 BEV가 x=[0,25] m, y=[-15,15] m, 0.05 m/pixel을 공유함
[ ] 두 토픽의 header timestamp가 동기화 허용범위 이내임
[ ] 두 토픽의 frame 기준이 base_link로 일관됨
[ ] /visualization/bev_overlay가 지속적으로 발행됨
[ ] RViz Image Display에서 Camera와 LiDAR가 중첩됨
[ ] Lidar3D-4 /waypoints가 base_link로 변환되어 합성 화면에 경로가 표시됨
[ ] 정지·직진·회전 구간에서 정합 상태가 유지됨
```

이 검증은 모델 추론 정확도와 별개로, 센서 BEV 정합·시간 동기·waypoint frame 변환이 통합 계약을 따르는지 확인한다.
