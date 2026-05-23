cmake_minimum_required(VERSION 3.16)
project(smarthome_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS
  core
  imgproc
  highgui
  cudaarithm
  cudaimgproc
  cudawarping
)
find_package(rosidl_default_generators REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR
  NvInfer.h
  PATHS
    /usr/include
    /usr/include/aarch64-linux-gnu
    /usr/local/include
)

find_library(TENSORRT_LIB
  nvinfer
  PATHS
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
)

if(NOT TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR "NvInfer.h not found")
endif()

if(NOT TENSORRT_LIB)
  message(FATAL_ERROR "nvinfer library not found")
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedTarget.msg"
  DEPENDENCIES builtin_interfaces std_msgs
)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

add_library(${PROJECT_NAME}_core SHARED
  src/pose_solver.cpp
  src/trt_detector.cpp
  src/detector.cpp
)

ament_target_dependencies(${PROJECT_NAME}_core
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  camera_info_manager
)

target_link_libraries(${PROJECT_NAME}_core
  ${OpenCV_LIBS}
  ${TENSORRT_LIB}
  CUDA::cudart
)

add_executable(vision_node
  src/vision_node.cpp
)

ament_target_dependencies(vision_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  camera_info_manager
)

target_link_libraries(vision_node
  ${PROJECT_NAME}_core
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(vision_node "${cpp_typesupport_target}")

install(TARGETS
  ${PROJECT_NAME}_core
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  vision_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_dependencies(rosidl_default_runtime)

ament_package()