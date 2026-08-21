# Open-loop 운영 절차

## 기본 실행

저장소 root에서 다음 순서로 실행한다. 아래는 InterFuser 예시이며, `interfuser` 대신 `tfv6`, `diffusiondrive_v2`, `sparsedrive_v2`를 넣어도 같은 절차로 실행된다. 첫 명령은 선택 모델 전용 환경을 준비하고, 두 번째 명령은 validator와 RViz를 시작한다. 세 번째 명령은 별도 terminal에서 기본 bag과 `/clock`을 재생한다.

```bash
./openloop setup interfuser
./openloop interfuser --rviz
./openloop bag
```

RViz가 필요 없으면 `./openloop interfuser`처럼 모델 이름만 사용한다. 다른 bag은 `./openloop bag --path /absolute/path/to/input.bag`으로 지정한다.

## 환경 관리

지원 host는 Ubuntu 20.04 x86_64와 ROS Noetic이며, PATH에서 정확히 `uv 0.11.14`를 찾을 수 있어야 한다. SparseDrive V2의 CUDA extension에 필요한 최소 CUDA 11.8 compiler는 setup이 프로젝트 `.toolchains/cuda-11.8`에 exact SHA-256 패키지로 준비한다. 시스템 CUDA 설치나 별도 `CUDA_HOME` 설정은 필요 없다.

활성 환경의 source, runtime asset, ROS overlay, Python package lock을 다시 검사하려면 다음 명령을 실행한다.

```bash
./openloop verify interfuser
```

활성 build를 보존하면서 이전 비활성 build를 삭제하려면 다음 명령을 실행한다.

```bash
./openloop prune interfuser
```

환경이 없거나 빠른 fingerprint가 달라진 경우 `run`은 ROS를 시작하지 않고 다음 복구 명령을 출력한다.

```text
MODEL environment is missing or stale.
Run: ./openloop setup MODEL
```

setup 오류는 실패한 phase와 prerequisite를 함께 표시한다. 새 환경을 강제로 만들 때는 `./openloop setup MODEL --recreate`를 사용한다.

## 모델 준비 범위

`interfuser`, `tfv6`, `diffusiondrive_v2`, `sparsedrive_v2` 네 모델의 Python version, hash lock, CUDA 정책, source commit과 setup/preflight 계약이 정의되어 있다. 네 모델 모두 setup에서 source/assets/dependency와 실제 model load를 검증한다. TFv6는 CUDA GPU를 요구하며, SparseDrive V2는 shipped camera calibration을 검증하고 프로젝트 로컬 CUDA 11.8 compiler로 extension을 빌드한다.

InterFuser checkpoint는 Drivora commit `25d56d2300f422d502c6549695b9e49efd5e5e00`이 가리키는 Google Drive object를 검증한 `third-party compatibility artifact`다. 기대 SHA256은 `39548dcfa56cdc06b8ccc52d7f6187a7d70ef36b05b7027db4d64a57ac3e30b2`이며, 파일이 없으면 setup이 기대 경로와 provenance를 출력한다.

## 고급 직접 launch

CLI가 전달하는 인자를 직접 제어해야 할 때만 ROS setup을 source한 terminal에서 절대 환경 root를 지정한다.

```bash
roslaunch Validator_pkg openloop_validator.launch \
  project_root:=/home/stier/E2E-OpenLoop-ROS-Validator \
  model_id:=interfuser \
  model_env_root:=/home/stier/E2E-OpenLoop-ROS-Validator/.venvs \
  show_rviz:=true
```

직접 launch는 환경을 생성하거나 검증하지 않는다. 먼저 `./openloop setup interfuser` 또는 `./openloop verify interfuser`를 실행한다.

고급 직접 launch에서는 `profile_file:=/absolute/profile.json`과 `record_dir:=/absolute/records`를 추가해 profile과 기록 경로를 바꿀 수 있다.
