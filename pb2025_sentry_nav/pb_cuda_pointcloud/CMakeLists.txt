cmake_minimum_required(VERSION 3.18)
project(pb_cuda_pointcloud LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PB_NAV_USE_CUDA "AUTO" CACHE STRING "CUDA backend mode: AUTO, ON, or OFF")
set(PB_CUDA_ARCHITECTURES "87" CACHE STRING "CUDA architectures for Jetson Orin NX")
set_property(CACHE PB_NAV_USE_CUDA PROPERTY STRINGS AUTO ON OFF)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

set(pb_cuda_sources src/pointcloud_accel.cpp)
set(pb_cuda_compile_defs PB_CUDA_POINTCLOUD_HAS_CUDA=0)
set(pb_cuda_enabled OFF)

if(NOT PB_NAV_USE_CUDA STREQUAL "OFF")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      list(APPEND pb_cuda_sources cuda/pointcloud_kernels.cu)
      set(pb_cuda_compile_defs PB_CUDA_POINTCLOUD_HAS_CUDA=1)
      set(pb_cuda_enabled ON)
      set(CMAKE_CUDA_ARCHITECTURES ${PB_CUDA_ARCHITECTURES})
      set(CMAKE_CUDA_STANDARD 14)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
      message(STATUS "pb_cuda_pointcloud: CUDA enabled for sm_${PB_CUDA_ARCHITECTURES}")
    elseif(PB_NAV_USE_CUDA STREQUAL "ON")
      message(FATAL_ERROR "PB_NAV_USE_CUDA=ON but CUDAToolkit was not found")
    endif()
  elseif(PB_NAV_USE_CUDA STREQUAL "ON")
    message(FATAL_ERROR "PB_NAV_USE_CUDA=ON but no CUDA compiler was found")
  endif()
endif()

add_library(${PROJECT_NAME} SHARED ${pb_cuda_sources})
target_compile_definitions(${PROJECT_NAME} PRIVATE ${pb_cuda_compile_defs})
if(pb_cuda_enabled)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${PB_CUDA_ARCHITECTURES})
endif()
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})
ament_target_dependencies(${PROJECT_NAME}
  livox_ros_driver2
  pcl_conversions
  rclcpp
  sensor_msgs
)

if(pb_cuda_enabled AND TARGET CUDA::cudart)
  target_link_libraries(${PROJECT_NAME} CUDA::cudart)
endif()

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(
  Eigen3
  livox_ros_driver2
  PCL
  pcl_conversions
  rclcpp
  sensor_msgs
)
if(pb_cuda_enabled)
  ament_export_dependencies(CUDAToolkit)
endif()

ament_package()
