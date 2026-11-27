# Ubuntu 22.04·Humble 실행 안내

## 최초 준비

이 프로젝트는 설치된 ROS2 Humble과 Ubuntu의 `/usr/bin/python3` 3.10을 사용합니다. 시스템 Python을 교체하거나 Windows의 `.venv`를 복사하지 않습니다. conda·다른 Python 가상환경 밖의 새 터미널에서 실행하는 편이 좋습니다.

```bash
git clone https://github.com/tjsong1234/Dolbat_Toyproject.git
cd Dolbat_Toyproject
./scripts/run.sh
```

`run.sh`가 의존성 검사, 필요한 apt 패키지 설치, colcon 빌드, launch를 차례로 수행합니다. 최초 apt 설치에는 인터넷과 sudo 권한이 필요합니다. ROS2 자체가 설치되지 않았으면 중단하며, 임의로 다른 배포판을 설치하지 않습니다.

`setup_ubuntu.sh`는 OpenCV·NumPy·YAML·Matplotlib·pytest·colcon·cv_bridge·ROS 메시지 생성 도구·FFmpeg를 apt로 준비합니다. `sudo pip`, Python 심볼릭 링크 교체, `.bashrc` 수정은 수행하지 않습니다.

## 업데이트

```bash
cd ~/Dolbat_Toyproject  # clone한 실제 위치로 이동
git pull --ff-only
./scripts/run.sh
```

소스 수정 후에는 자동 빌드를 유지하세요. `--no-build`는 같은 소스를 다시 실행할 때만 사용합니다. 로컬 수정 때문에 pull이 거부되면 먼저 `git status`로 변경을 확인하세요. 실행 스크립트는 로컬 파일을 삭제하거나 강제로 덮어쓰지 않습니다.

## 자주 확인할 내용

### 창이 뜨지 않는다

기본은 저장만 하는 headless 실행입니다. 실제 Ubuntu 데스크톱의 터미널에서 `./scripts/run.sh --display`로 실행하세요. 원격 SSH나 CI처럼 화면 서버가 없는 환경에서는 headless를 유지합니다.

### 의존성 설치가 실패한다

네트워크와 apt 저장소 상태를 확인하고 `./scripts/setup_ubuntu.sh`를 다시 실행합니다. 설치된 ROS2 배포판은 `/opt/ros/humble`이어야 합니다. 다른 ROS2 배포판이나 다른 Python의 패키지를 섞어 해결하지 마세요.

### 이전 빌드를 찾지 못하거나 새 메시지가 반영되지 않는다

`./scripts/run.sh`를 `--no-build` 없이 실행하세요. 소스가 바뀌면 colcon이 해당 패키지를 다시 빌드합니다. 메시지 패키지는 노드 패키지보다 먼저 빌드됩니다.

### 원본 영상 경로 오류

기본 영상은 저장소의 `data/dolbat_toyproject.mp4`입니다. 원본 파일의 SHA-256으로 입력을 확인하므로 다른 주행영상으로 대체하지 마세요. 전체 clone/pull에 이 파일이 포함되며 Git LFS나 별도 다운로드는 필요하지 않습니다.

### 노드 통신 또는 실행 종료 문제

스크립트는 기본적으로 `ROS_LOCALHOST_ONLY=1`, `rmw_fastrtps_cpp`를 현재 프로세스에 설정합니다. 모든 단계가 같은 컴퓨터에서 실행되도록 설계한 기본값입니다. 각 실행은 새 출력 폴더를 사용하고 `summary.json`과 `failure.txt`를 확인하세요. 실패했는데 출력 영상의 일부만 보고 성공으로 판단하지 않습니다.

### 성능 측정

`replay_rate:=1.0`은 원본 재생 속도에 맞추므로 빠른 컴퓨터에서도 입력률이 24FPS로 제한됩니다. `replay_rate:=0.0`은 완료 ACK의 제한 안에서 처리하므로 처리량을 측정할 때 사용합니다. 출력 MP4는 두 경우 모두 원본 시간축의 24FPS입니다. 평균 처리량과 개별 프레임 지연, 실제 차량 제어 안정성은 서로 다른 지표입니다.

## 확인한 공식 자료

- [ROS2 Humble Ubuntu deb 설치](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- [ROS2 인터페이스 패키지](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html)
- [ROS2 QoS](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html)

이 저장소는 사용자 프로젝트 코드를 담으며 별도의 오픈소스 라이선스 권한을 새로 부여하지 않습니다(`UNLICENSED`).
