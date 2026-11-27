#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$ROOT_DIR/ros2_ws/install/setup.bash"
set -u
cd -- "$ROOT_DIR/ros2_ws"
/usr/bin/colcon test --packages-select dolbat_core dolbat_pipeline \
  --event-handlers console_direct+ --return-code-on-test-failure
/usr/bin/colcon test-result --verbose
