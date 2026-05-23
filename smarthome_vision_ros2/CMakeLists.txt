cmake_minimum_required(VERSION 3.16)
project(smarthome_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SMARTHOME_VISION_WITH_OPENVINO "Build the OpenVINO inference backend" ON)
option(SMARTHOME_VISION_WITH_TENSORRT "Build the TensorRT inference backend" OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(camera_info_manager REQUIRED)

set(SMARTHOME_VISION_OPENCV_COMPONENTS
  core
  calib3d
  imgproc
  highgui
  videoio
)

if(SMARTHOME_VISION_WITH_TENSORRT)
  list(APPEND SMARTHOME_VISION_OPENCV_COMPONENTS
    cudaarithm
    cudaimgproc
    cudawarping
  )
endif()

find_package(OpenCV REQUIRED COMPONENTS ${SMARTHOME_VISION_OPENCV_COMPONENTS})
find_package(rosidl_default_generators REQUIRED)

set(SMARTHOME_VISION_BACKEND_SOURCES)
set(SMARTHOME_VISION_BACKEND_LIBS)
set(SMARTHOME_VISION_BACKEND_INCLUDE_DIRS)
set(SMARTHOME_VISION_BACKEND_DEFINES)

if(SMARTHOME_VISION_WITH_OPENVINO)
  find_package(OpenVINO REQUIRED)
  list(APPEND SMARTHOME_VISION_BACKEND_SOURCES src/openvino_detector.cpp)
  list(APPEND SMARTHOME_VISION_BACKEND_LIBS openvino::runtime)
  list(APPEND SMARTHOME_VISION_BACKEND_DEFINES SMARTHOME_VISION_WITH_OPENVINO)
endif()

if(SMARTHOME_VISION_WITH_TENSORRT)
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

  list(APPEND SMARTHOME_VISION_BACKEND_SOURCES src/trt_detector.cpp)
  list(APPEND SMARTHOME_VISION_BACKEND_LIBS ${TENSORRT_LIB} CUDA::cudart)
  list(APPEND SMARTHOME_VISION_BACKEND_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
  list(APPEND SMARTHOME_VISION_BACKEND_DEFINES SMARTHOME_VISION_WITH_TENSORRT)
endif()

if(NOT SMARTHOME_VISION_WITH_OPENVINO AND NOT SMARTHOME_VISION_WITH_TENSORRT)
  message(FATAL_ERROR "At least one inference backend must be enabled")
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedTarget.msg"
  DEPENDENCIES builtin_interfaces std_msgs
)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${SMARTHOME_VISION_BACKEND_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_core SHARED
  src/pose_solver.cpp
  src/detector.cpp
  ${SMARTHOME_VISION_BACKEND_SOURCES}
)

target_compile_definitions(${PROJECT_NAME}_core PRIVATE ${SMARTHOME_VISION_BACKEND_DEFINES})

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
  ${SMARTHOME_VISION_BACKEND_LIBS}
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
