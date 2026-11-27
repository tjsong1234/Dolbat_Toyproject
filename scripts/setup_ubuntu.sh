#!/usr/bin/env bash
# Install this project's dependencies; retain the existing Humble/Python setup.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then CHECK_ONLY=1; shift; fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--check-only]" >&2
  exit 2
fi
if [[ ! -f /etc/os-release ]]; then
  echo "Ubuntu 22.04 is required." >&2
  exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 22.04 ]]; then
  echo "Supported environment: Ubuntu 22.04 + ROS2 Humble; detected ${PRETTY_NAME:-unknown}." >&2
  exit 1
fi
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS2 Humble must already be installed at /opt/ros/humble." >&2
  echo "See https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html" >&2
  exit 1
fi
if ! /usr/bin/python3 -c 'import sys; assert sys.version_info[:2] == (3, 10)' ; then
  echo "Use Ubuntu's /usr/bin/python3 (3.10); do not replace system Python." >&2
  exit 1
fi

# ROS setup files legitimately read unset environment variables.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u
export PYTHONNOUSERSITE=1

check_dependencies() {
  /usr/bin/python3 - <<'PY'
import importlib
modules = ('rclpy', 'cv_bridge', 'cv2', 'numpy', 'yaml', 'matplotlib', 'pytest', 'colcon_core')
missing = []
for name in modules:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit('Missing project dependencies: ' + ', '.join(missing))
PY
  test -x /usr/bin/colcon
  command -v cmake >/dev/null
  command -v c++ >/dev/null
  command -v ffmpeg >/dev/null
  command -v ffprobe >/dev/null
  ros2 pkg prefix rosidl_default_generators >/dev/null
  ros2 pkg prefix ament_cmake >/dev/null
  ros2 pkg prefix launch_ros >/dev/null
  ros2 pkg prefix geometry_msgs >/dev/null
  ros2 pkg prefix nav_msgs >/dev/null
}

if [[ $CHECK_ONLY -eq 1 ]]; then
  check_dependencies
  exit 0
fi

if [[ $EUID -eq 0 ]]; then
  SUDO=()
else
  command -v sudo >/dev/null || { echo "sudo is required to install apt dependencies." >&2; exit 1; }
  SUDO=(sudo)
fi
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y --no-install-recommends \
  build-essential cmake git \
  python3-colcon-common-extensions python3-rosdep python3-setuptools \
  python3-opencv python3-numpy python3-matplotlib python3-yaml python3-pytest \
  ffmpeg \
  ros-humble-cv-bridge ros-humble-launch ros-humble-launch-ros \
  ros-humble-rosidl-default-generators ros-humble-rmw-fastrtps-cpp \
  ros-humble-std-msgs ros-humble-sensor-msgs ros-humble-geometry-msgs ros-humble-nav-msgs
check_dependencies
echo "Project dependencies are ready. Run: $ROOT_DIR/scripts/run.sh"
