from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution([
        FindPackageShare("robot_serial_comm"),
        "config",
        "robot_serial_comm.yaml",
    ])

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=config_file,
        description="Full path to robot_serial_comm parameters",
    )

    robot_serial_comm_node = Node(
        package="robot_serial_comm",
        executable="robot_serial_comm_node",
        name="robot_serial_comm",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription([
        params_file_arg,
        robot_serial_comm_node,
    ])
