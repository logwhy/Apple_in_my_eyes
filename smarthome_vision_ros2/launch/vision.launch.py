from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    inference_backend = LaunchConfiguration("inference_backend")
    openvino_device = LaunchConfiguration("openvino_device")
    include_serial = LaunchConfiguration("include_serial")
    use_cam2image = LaunchConfiguration("use_cam2image")
    use_local_camera = LaunchConfiguration("use_local_camera")
    camera_device_id = LaunchConfiguration("camera_device_id")
    show_debug = LaunchConfiguration("show_debug")
    log_debug_result = LaunchConfiguration("log_debug_result")
    debug_view_scale = LaunchConfiguration("debug_view_scale")
    use_mode_control = LaunchConfiguration("use_mode_control")
    test_mode = LaunchConfiguration("test_mode")

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
        ),
        condition=IfCondition(include_serial),
    )

    cam_node = Node(
        package="image_tools",
        executable="cam2image",
        name="cam2image",
        output="log",
        parameters=[
            {"device_id": ParameterValue(camera_device_id, value_type=int)}
        ],
        remappings=[
            ("image", "/image_raw")
        ],
        condition=IfCondition(use_cam2image),
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
                "use_local_camera": ParameterValue(use_local_camera, value_type=bool),
                "camera_device_id": ParameterValue(camera_device_id, value_type=int),
                "use_mode_control": ParameterValue(use_mode_control, value_type=bool),
                "use_test_mode": True,
                "test_mode": ParameterValue(test_mode, value_type=int),
                "show_debug": ParameterValue(show_debug, value_type=bool),
                "log_debug_result": ParameterValue(log_debug_result, value_type=bool),
                "debug_view_scale": ParameterValue(debug_view_scale, value_type=int),
            }
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument("inference_backend", default_value="openvino"),
        DeclareLaunchArgument("openvino_device", default_value="CPU"),
        DeclareLaunchArgument("include_serial", default_value="false"),
        DeclareLaunchArgument("use_cam2image", default_value="false"),
        DeclareLaunchArgument("use_local_camera", default_value="true"),
        DeclareLaunchArgument("camera_device_id", default_value="0"),
        DeclareLaunchArgument("show_debug", default_value="true"),
        DeclareLaunchArgument("log_debug_result", default_value="true"),
        DeclareLaunchArgument("debug_view_scale", default_value="2"),
        DeclareLaunchArgument("use_mode_control", default_value="false"),
        DeclareLaunchArgument("test_mode", default_value="1"),
        robot_serial_comm_launch,
        cam_node,
        vision_node
    ])
