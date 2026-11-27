#!/usr/bin/env bash
# A pull is followed by dependency checking, a local build, then one launch.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$ROOT_DIR/ros2_ws"
VIDEO="$ROOT_DIR/data/dolbat_toyproject.mp4"
OUTPUT="$ROOT_DIR/outputs/$(date +%Y%m%d_%H%M%S)"
SHOW_DISPLAY=false
REPLAY_RATE=1.0
MAX_IN_FLIGHT=4
INSTALL_MISSING=1
BUILD=1
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --no-install) INSTALL_MISSING=0 ;;
    --no-build) BUILD=0 ;;
    --display) SHOW_DISPLAY=true ;;
    --headless) SHOW_DISPLAY=false ;;
    --help|-h)
      cat <<'HELP'
Usage: scripts/run.sh [--display|--headless] [--no-install] [--no-build]
                      [video:=/path/to/original.mp4] [output:=/path/to/new/output]
                      [replay_rate:=1.0] [max_in_flight:=4]
replay_rate 1.0 follows the recording's 24 FPS; 0.0 runs as fast as credits allow.
Missing dependencies are installed through apt on the first run (sudo required).
--no-install checks dependencies without installing; --no-build reuses a build.
HELP
      exit 0 ;;
    video:=*) VIDEO="${arg#video:=}" ;;
    output:=*) OUTPUT="${arg#output:=}" ;;
    display:=*) SHOW_DISPLAY="${arg#display:=}" ;;
    replay_rate:=*) REPLAY_RATE="${arg#replay_rate:=}" ;;
    max_in_flight:=*) MAX_IN_FLIGHT="${arg#max_in_flight:=}" ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

if ! bash "$ROOT_DIR/scripts/setup_ubuntu.sh" --check-only; then
  if [[ $INSTALL_MISSING -eq 0 ]]; then
    echo "Dependencies are not ready. Run scripts/setup_ubuntu.sh first." >&2
    exit 1
  fi
  echo "Preparing missing Ubuntu project dependencies (sudo may request your password)."
  bash "$ROOT_DIR/scripts/setup_ubuntu.sh"
fi
if [[ ! -f "$VIDEO" ]]; then
  echo "Original video does not exist: $VIDEO" >&2
  exit 1
fi
if [[ "$SHOW_DISPLAY" != true && "$SHOW_DISPLAY" != false ]]; then
  echo "display must be true or false." >&2
  exit 2
fi

# These changes apply to this script, not the user's shell or other projects.
export PYTHONNOUSERSITE=1
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u
cd -- "$WORKSPACE"
if [[ $BUILD -eq 1 ]]; then
  /usr/bin/colcon build --symlink-install \
    --packages-up-to dolbat_pipeline \
    --event-handlers console_direct+ \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
fi
if [[ ! -f "$WORKSPACE/install/setup.bash" ]]; then
  echo "No built workspace found. Run without --no-build." >&2
  exit 1
fi
set +u
# shellcheck disable=SC1091
source "$WORKSPACE/install/setup.bash"
set -u
exec ros2 launch dolbat_pipeline dolbat.launch.py \
  "video:=$VIDEO" "output:=$OUTPUT" "display:=$SHOW_DISPLAY" \
  "replay_rate:=$REPLAY_RATE" "max_in_flight:=$MAX_IN_FLIGHT" "${EXTRA_ARGS[@]}"
