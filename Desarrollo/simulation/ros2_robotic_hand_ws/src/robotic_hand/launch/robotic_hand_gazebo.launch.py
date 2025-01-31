import os
import xacro  # ✅ Added Xacro import

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

# Get package path
packagepath = get_package_share_directory('robotic_hand')

# File paths
xacro_file_path = os.path.join(packagepath, 'urdf', 'ros2_control', 'gazebo', 'robotic_hand.urdf.xacro')
world_file_path = os.path.join(packagepath, 'worlds', 'empty.sdf')
config_file_path = os.path.join(packagepath, 'config', 'hand_controllers.yaml')

# ✅ Convert Xacro to URDF
robot_desc = xacro.process_file(xacro_file_path).toxml()

def generate_launch_description():
    # Launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='true',
        description='Use simulation time if true'
    )

    declare_world = DeclareLaunchArgument(
        name='world',
        default_value=world_file_path,
        description='Full path to the world file'
    )

    # Gazebo launch
    start_gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
        )),
        launch_arguments={'gz_args': [world_file_path, ' -r -v4']}.items()
    )

    # Robot to Gazebo
    spawn_robot_cmd = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-string', robot_desc, '-x', '0', '-y', '0', '-z', '0.05', '-name', 'robotic_hand']
    )

    # ROS 2 bridge node
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': config_file_path}]
    )

    # ✅ Robot State Publisher with Correct Robot Description
    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': robot_desc},
        ]
    )

    """# Arm controller
    load_arm_controller_cmd = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller'],
        output='screen'
    )

    # Gripper controller
    load_gripper_controller_cmd = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['grip_controller'],
        output='screen'
    )"""

    # Launch description
    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_world)

    # Add actions
    ld.add_action(start_gazebo_cmd)
    ld.add_action(spawn_robot_cmd)
    """ ld.add_action(bridge_node)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(load_arm_controller_cmd)
    ld.add_action(load_gripper_controller_cmd)"""

    return ld
