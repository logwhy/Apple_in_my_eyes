cmake_minimum_required(VERSION 3.16)
project(robot_serial_comm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(smarthome_vision REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME}_core SHARED
  src/crc16.cpp
  src/serial_bridge.cpp
)

add_executable(robot_serial_comm_node
  src/robot_serial_comm_node.cpp
)

ament_target_dependencies(robot_serial_comm_node
  rclcpp
  geometry_msgs
  std_msgs
  smarthome_vision
)

target_link_libraries(robot_serial_comm_node
  ${PROJECT_NAME}_core
)

install(TARGETS
  ${PROJECT_NAME}_core
  robot_serial_comm_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_package()
