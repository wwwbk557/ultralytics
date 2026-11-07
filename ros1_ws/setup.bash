#!/usr/bin/env bash
# Usage (inside the r1 container): source /home/wbk/ultralytics_2027/ros1_ws/setup.bash
source /opt/ros/noetic/setup.bash
_ROS1_WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${_ROS1_WS_DIR}/devel/setup.bash" ]; then
  source "${_ROS1_WS_DIR}/devel/setup.bash"
else
  echo "warning: ${_ROS1_WS_DIR}/devel/setup.bash not found, run ./build.sh first" >&2
fi
