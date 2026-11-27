# ROS2 Humble 전환 — 중단 시점 인수인계

기록일: 2026-11-27 (Asia/Seoul)

**상태: 사용자 요청으로 중단. 현재 ROS2 코드는 실행 가능한 완성본이 아니다. 커밋·푸시는 하지 않았다.**

## 사용자 요청과 확정 범위

Ubuntu 22.04 듀얼부팅에 ROS2 Humble이 설치되어 있다. 기존 1~4주차 기능을 ROS2 노드로 나누고, 기존 ROI·BEV·차선 검출·B-spline 경로·Pure Pursuit 결과를 유지한다. 사용자는 이어서 아래 GitHub 저장소에 실제 구현을 커밋·푸시하고, Ubuntu에서 clone/pull 후 실행할 수 있도록 만들 것을 승인했다.

- 저장소: https://github.com/tjsong1234/Dolbat_Toyproject
- 로컬: `C:/Users/tjson/Desktop/자율주행/Dolbat_Toyproject`
- 실제 차량 구동, 실시간 카메라 연결, 딥러닝 도입·차선 알고리즘 재설계는 범위 밖이다.
- 기존 Windows 프로젝트와 제출물은 보존한다.

## 실제로 완료한 작업

1. GitHub 저장소가 비어 있고 현재 계정에 ADMIN 권한이 있음을 확인했다. GitHub Actions도 활성화되어 있었다. 저장소를 로컬로 clone했고 커밋 작성자·줄바꿈 설정은 이 저장소에만 지정했다.
2. 원본 `data/dolbat_toyproject.mp4`(2,852,610bytes)를 복사했다. Git LFS 없이 저장소에 포함할 수 있는 크기다. 영상 SHA-256은 `c99bd02d43f12dbcd093b808ef7bdc6c85d17c3b5620808213c5e2e2e14cc66f`다.
3. `dolbat_interfaces`에 6개 메시지와 CMake/package 메타데이터 초안을 작성했다. 메시지는 VideoFrame, PreprocessedFrame, LaneCurve, LaneFrame, PathFrame, ControlFrame이다.
4. `dolbat_pipeline/dolbat_pipeline/joining.py`에 프레임 번호·타임스탬프 기반의 제한된 캐시 결합 초안을 작성하고 `test/test_joining.py`를 작성했다. **아직 이 테스트를 실행하지 않았다.**
5. `dolbat_pipeline/dolbat_pipeline/visualizer.py`에 4개 스트림 결합, 원본/Sliding Window/경로·LP/조향 화면, 영상·CSV 저장과 프레임 완료 ACK 초안을 작성했다. **아직 구현되지 않은 Runtime/common/codecs를 import하므로 현재 실행할 수 없다.**
6. `validation/`에 이전 Windows 최종 결과에서 수치 회귀 기준을 만들었다. `reference_control.csv`, `reference_paths.npz`(564×160×2, 후륜 기준 m), `reference_provenance.json`이다. 독립 정답 차선 라벨이 아니라 기존 구현 결과의 회귀 기준이다.
7. `scripts/verify_ros_run.py` 초안을 작성했다. 실제 ROS2 출력이 생긴 뒤 564프레임·경로·LP·조향·고정 축·MP4 전체 디코딩을 검사하는 용도이며 **아직 실행하지 않았다.**
8. 확정 의도를 `docs/intent/dolbat-ros2-humble.md`에 저장했다.

## Git 상태

- 로컬 브랜치: `main`.
- 상태: `No commits yet on main...origin/main [gone]`.
- `data/`, `docs/`, `ros2_ws/`, `scripts/`, `validation/`은 모두 미커밋 파일이다.
- GitHub에 코드 커밋·푸시·워크플로 실행을 하지 않았다. 현재 GitHub에서 pull할 완성 패키지는 없다.
- GitHub 인증 정보나 토큰을 소스에 기록하지 않았다.

## 아직 생성되지 않았거나 미완료인 것

- `dolbat_core` 패키지 자체: 아직 디렉터리/파일이 생성되지 않았다.
- 기존 순수 Python 모듈·설정·환산 자료의 패키지 이식과 Python 3.10 호환 처리.
- `dolbat_core.runtime.Runtime` 구현.
- 입력·전처리·Sliding Window·경로·Pure Pursuit 노드, ROS 메시지 encode/decode, 공통 QoS.
- `dolbat_pipeline`의 setup.py/setup.cfg/package.xml/resource 및 launch 파일.
- Ubuntu 의존성 설치·자동 빌드·실행 스크립트, README, .gitignore/.gitattributes.
- GitHub Actions 워크플로 파일.
- ROS2 인터페이스 생성, colcon build/test, ROS2 전체 영상 실행 및 회귀 검증.
- 커밋·푸시, CI 결과 확인 및 Ubuntu 사용자 실행 안내.

## 재개 시 사용할 설계 합의

노드는 `video_source → preprocessing(1주차) → sliding_window(2주차) → path_planning(3주차) → pure_pursuit(4주차)`이며 visualization이 원본/디버그/경로/제어를 결합한다. 차선 검출 이후에는 큰 영상 묶음 대신 좌표와 상태를 전달한다.

- 토픽 예정: `/dolbat/video`, `/dolbat/preprocessed`, `/dolbat/sliding_debug`, `/dolbat/lanes`, `/dolbat/path`, `/dolbat/control`.
- 완료 ACK: `/dolbat/frame_done`, `std_msgs/UInt32`로 frame_index 전달.
- 입력 중첩 프레임 수 4, 메시지 QoS reliable/transient_local/keep_last 8을 검토 중이다. 실제 Linux 동작은 검증하지 않았다.
- 원본 입력과 각 단계의 frame_index/source_fps/header stamp를 그대로 전달한다. 모든 프레임의 ACK 뒤 EOS를 전달하고 시각화 노드가 저장을 마치면 launch 전체를 종료하는 구조다.
- `replay_rate`는 배속: 1.0은 원본 시간, 0.0은 완료 ACK 범위에서 최대 처리 속도로 동작하는 방안을 정했다.
- 후륜 기준 표준 경로 `/dolbat/path_rear`(nav_msgs/Path)와 LP `/dolbat/lookahead_point`(geometry_msgs/PointStamped)도 검토 대상으로 전달했다. 구현은 아직 없다. 실제 액추에이터 명령 토픽은 만들지 않는다.

Runtime 예정 API: `Runtime(config_dir=None, fps=24, image_size=(640,480))`, 속성 `.base/.geo/.profile/.week/.calibration/.mapping/.pre/.detector/.planner/.control_config/.control_settings/.matrix/.inverse/.geometry/.translation`, 메서드 `.plan(lanes)`는 기존 plan+partial attachment+고정 축 설정을 유지한다. 이 API는 초안 연결을 위한 합의이며 현재 존재하는 코드가 아니다.

## 검증 및 환경 주의

- 현재 컴퓨터는 Windows이며 Docker는 발견되지 않았고 WSL Linux 배포판도 사용할 수 없었다. 사용자의 듀얼부팅 Ubuntu를 현재 세션에서 실행하지 않았다.
- GitHub CI에서 Ubuntu 22.04·Humble 컨테이너를 이용해 실제 ROS2 빌드/실행을 검증하려던 단계다. 아직 워크플로 파일조차 생성하지 않았다.
- Humble의 Python 환경과 기존 Python 3.12 프로젝트의 NumPy 버전 차이를 확인해야 한다. 예를 들어 `metric_mapping.py`의 `np.trapezoid`는 Ubuntu 구버전 NumPy 대응이 필요하다. 시스템 Python 교체나 전역 pip 설치로 해결하지 않는다.
- 이전 Windows 최적화 실행 25.3FPS 및 201개 테스트 통과를 ROS2 성능/빌드 검증으로 재사용하면 안 된다.
- 최초 의존성 준비와 이후 pull/자동 재빌드·실행을 구분해야 한다. 아직 작성하지 않은 스크립트를 실행하라고 안내하지 않는다.

## 중단 조치와 다음 시작점

사용자의 "잠시멈춰줘", "현재단계기록하고중단" 요청에 따라 병렬 작업자 `ros_core_port`, `ros_stage_nodes`, `ros_ubuntu_delivery`를 모두 interrupt했다. 이 문서와 파일 현황 스냅샷을 기록한 뒤 추가 구현·테스트·커밋·푸시는 진행하지 않는다.

재개 시 우선 실제 파일과 Git 상태를 다시 확인한다. 위 초안과 API를 검토한 후 core 이식 → 노드/코덱/launch 통합 → 로컬 가능한 검사 → GitHub 게시 및 실제 Humble CI 검증 순서로 남은 작업을 이어갈 수 있다.
