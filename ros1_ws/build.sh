#!/usr/bin/env bash
# Build the ROS1 workspace from inside the `r1` container (or any ROS Noetic shell).
set -e
source /opt/ros/noetic/setup.bash
cd "$(dirname "$0")"
catkin_make -DCMAKE_BUILD_TYPE=Release "$@"
