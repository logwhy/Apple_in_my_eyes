from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    inference_backend = LaunchConfiguration("inference_backend")
    openvino_device = LaunchConfiguration("openvino_device")

    params_file = os.path.join(
        get_package_share_directory("smarthome_vision"),
        "config",
        "vision.yaml"
    )

    robot_serial_comm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("robot_serial_comm"),
                "launch",
                "robot_serial_comm.launch.py",
            )
        )
    )

    cam_node = Node(
        package="image_tools",
        executable="cam2image",
        name="cam2image",
        output="log",
        parameters=[
            {"device_id": 0}
        ],
        remappings=[
            ("image", "/image_raw")
        ]
    )

    vision_node = Node(
        package="smarthome_vision",
        executable="vision_node",
        name="smarthome_vision_node",
        output="screen",
        parameters=[
            params_file,
            {
                "inference_backend": inference_backend,
                "openvino_device": openvino_device,
                "use_test_mode": True,
                "test_mode": 1,
                "show_debug": True
            }
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument("inference_backend", default_value="openvino"),
        DeclareLaunchArgument("openvino_device", default_value="CPU"),
        robot_serial_comm_launch,
        cam_node,
        vision_node
    ])
