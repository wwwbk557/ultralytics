cmake_minimum_required(VERSION 3.10.2)
project(hk_pose_openvino)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)

find_package(catkin REQUIRED COMPONENTS
  nodelet
  pluginlib
  roscpp
  sensor_msgs
  cv_bridge
  image_transport
  roslib
  dynamic_reconfigure
)
generate_dynamic_reconfigure_options(
  cfg/PoseConfig.cfg
)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# OpenVINO
#
# Do NOT use find_package(OpenVINO): its CMake config calls
#   add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
# for the whole directory.  OpenVINO binaries come in both flavours
# (system 2024.1: ABI=1, conda/PyPI: ABI=0) while ROS Noetic/pluginlib is
# always ABI=1.  Locate the runtime manually and compile only
# src/openvino_detector.cpp with the ABI reported by OpenVINOConfig.cmake;
# the nodelet TU stays ABI=1.
#
# Override the search with -DOpenVINO_DIR=/path/to/openvino/cmake
# ---------------------------------------------------------------------------
if(OpenVINO_DIR)
  get_filename_component(OpenVINO_ROOT "${OpenVINO_DIR}/.." ABSOLUTE)
else()
  # Prefer the newest system installation (e.g. /opt/intel/openvino_2024.6.0),
  # then fall back to a conda/PyPI package.
  file(GLOB _ov_system_dirs
    "/opt/intel/openvino*/runtime/cmake"
    "/opt/intel/openvino*/cmake"
    "/opt/intel/openvino_*/runtime/cmake"
  )
  # Newest first: this machine may have openvino_2022 and openvino_2024.6.0.
  list(SORT _ov_system_dirs)
  list(REVERSE _ov_system_dirs)
  file(GLOB _ov_user_dirs
    "$ENV{HOME}/miniconda3/envs/*/lib/python*/site-packages/openvino/cmake"
    "$ENV{HOME}/miniconda3/envs/*/lib/python*/site-packages/openvino-*/openvino/cmake"
    "/home/*/miniconda3/envs/*/lib/python*/site-packages/openvino/cmake"
    "/home/*/miniconda3/envs/*/lib/python*/site-packages/openvino-*/openvino/cmake"
    "$ENV{CONDA_PREFIX}/lib/python*/site-packages/openvino/cmake"
    "$ENV{CONDA_PREFIX}/lib/python*/site-packages/openvino-*/openvino/cmake"
  )
  set(_ov_cmake_dirs ${_ov_system_dirs} ${_ov_user_dirs})
  foreach(_dir IN LISTS _ov_cmake_dirs)
    if(EXISTS "${_dir}/OpenVINOConfig.cmake")
      get_filename_component(OpenVINO_ROOT "${_dir}/.." ABSOLUTE)
      message(STATUS "hk_pose_openvino: found OpenVINO at ${OpenVINO_ROOT}")
      break()
    endif()
  endforeach()
endif()

if(NOT OpenVINO_ROOT OR NOT EXISTS "${OpenVINO_ROOT}/include/openvino/openvino.hpp")
  message(FATAL_ERROR
    "OpenVINO runtime not found.\n"
    "Pass -DOpenVINO_DIR=/path/to/openvino/cmake (the directory that contains\n"
    "OpenVINOConfig.cmake), e.g.\n"
    "  catkin_make -DOpenVINO_DIR=/opt/intel/openvino_2024/runtime/cmake")
endif()

# OpenVINO's official binaries are built either with the old libstdc++ ABI
# (conda/PyPI: _GLIBCXX_USE_CXX11_ABI=0) or the new one (system 2024.1: 1).
# Read the value from the config and compile only the detector TU with it;
# ROS/pluginlib must stay ABI=1.
set(OpenVINO_CXX11_ABI "1")
if(EXISTS "${OpenVINO_ROOT}/cmake/OpenVINOConfig.cmake")
  file(STRINGS "${OpenVINO_ROOT}/cmake/OpenVINOConfig.cmake" _ov_abi_lines
       REGEX "OpenVINO_GLIBCXX_USE_CXX11_ABI")
  foreach(_line IN LISTS _ov_abi_lines)
    if(_line MATCHES "OpenVINO_GLIBCXX_USE_CXX11_ABI[ \t]+\"?([01])\"?")
      set(OpenVINO_CXX11_ABI "${CMAKE_MATCH_1}")
    endif()
  endforeach()
endif()

file(GLOB _ov_libs
  "${OpenVINO_ROOT}/libs/libopenvino.so*"
  "${OpenVINO_ROOT}/lib/libopenvino.so*"
  "${OpenVINO_ROOT}/lib/intel64/libopenvino.so*"
  "${OpenVINO_ROOT}/runtime/lib/intel64/libopenvino.so*"
)
if(NOT _ov_libs)
  message(FATAL_ERROR "libopenvino.so not found under ${OpenVINO_ROOT}")
endif()
list(SORT _ov_libs)
list(GET _ov_libs 0 OpenVINO_LIBRARY)
get_filename_component(OpenVINO_LIB_DIR "${OpenVINO_LIBRARY}" DIRECTORY)

# libopenvino and its plugins need oneTBB, but the system OpenVINO install
# ships it in runtime/3rdparty/tbb/lib with no RPATH.  Link it explicitly
# (before libopenvino, so it is already loaded when libopenvino resolves its
# own NEEDED entry) and put its directory into RUNPATH.  Without this,
# `nodelet load` fails with "libtbb.so.12: cannot open shared object file".
set(OpenVINO_TBB_LIBRARY "")
foreach(_tbb_dir
    "${OpenVINO_ROOT}/3rdparty/tbb/lib"
    "${OpenVINO_ROOT}/runtime/3rdparty/tbb/lib"
    "${OpenVINO_LIB_DIR}")
  if(EXISTS "${_tbb_dir}/libtbb.so.12")
    set(OpenVINO_TBB_LIBRARY "${_tbb_dir}/libtbb.so.12")
    break()
  endif()
endforeach()

set(OpenVINO_RPATH_DIRS "${OpenVINO_LIB_DIR}")
if(OpenVINO_TBB_LIBRARY)
  get_filename_component(_tbb_dir "${OpenVINO_TBB_LIBRARY}" DIRECTORY)
  list(APPEND OpenVINO_RPATH_DIRS "${_tbb_dir}")
endif()
list(REMOVE_DUPLICATES OpenVINO_RPATH_DIRS)

message(STATUS "hk_pose_openvino: OpenVINO include ${OpenVINO_ROOT}/include")
message(STATUS "hk_pose_openvino: OpenVINO library ${OpenVINO_LIBRARY}")
message(STATUS "hk_pose_openvino: OpenVINO TBB ${OpenVINO_TBB_LIBRARY}")
message(STATUS "hk_pose_openvino: OpenVINO libstdc++ ABI=${OpenVINO_CXX11_ABI}")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES hk_pose_openvino_nodelet
  CATKIN_DEPENDS nodelet pluginlib roscpp sensor_msgs cv_bridge image_transport roslib dynamic_reconfigure
  DEPENDS OpenCV
)

add_library(hk_pose_openvino_nodelet
  src/openvino_detector.cpp
  src/pose_nodelet.cpp
)

add_dependencies(hk_pose_openvino_nodelet ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(hk_pose_openvino_nodelet hk_pose_openvino_gencfg)

target_include_directories(hk_pose_openvino_nodelet
  PUBLIC
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${OpenVINO_ROOT}/include
)

target_link_libraries(hk_pose_openvino_nodelet
  PUBLIC
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${OpenVINO_TBB_LIBRARY}
    ${OpenVINO_LIBRARY}
    Threads::Threads
)
if(OpenVINO_TBB_LIBRARY)
  # Keep the TBB NEEDED entry even though the nodelet itself does not call it.
  target_link_options(hk_pose_openvino_nodelet PRIVATE "-Wl,--no-as-needed")
endif()

# The only two lines that matter for the ABI split.
set_source_files_properties(src/openvino_detector.cpp PROPERTIES
  COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=${OpenVINO_CXX11_ABI};OV_THREAD=OV_THREAD_TBB")
set_source_files_properties(src/pose_nodelet.cpp PROPERTIES
  COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=1")

set_target_properties(hk_pose_openvino_nodelet PROPERTIES
  BUILD_RPATH "${OpenVINO_RPATH_DIRS}"
  INSTALL_RPATH "${OpenVINO_RPATH_DIRS}"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(TARGETS hk_pose_openvino_nodelet
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch models test_images
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/publish_test_image.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
