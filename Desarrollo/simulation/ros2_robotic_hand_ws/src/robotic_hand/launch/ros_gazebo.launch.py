import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # Package and file paths
    pkg_name = 'robotic_hand'
    file_subpath = 'description/robotic_hand.urdf.xacro'
    world_file = os.path.join(get_package_share_directory(pkg_name), 'worlds', 'empty.sdf')  # Ensure this world file exists

    # Process URDF
    xacro_file = os.path.join(get_package_share_directory(pkg_name), file_subpath)
    robot_description_raw = xacro.process_file(xacro_file).toxml()

    # **Launch Gazebo**
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'{world_file} -r -v4'}.items()
    )

    # **Robot State Publisher** (publishes TF)
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_raw, 'use_sim_time': True}]
    )

    # **Spawn the Robot in Gazebo**
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-string', robot_description_raw, '-x', '0', '-y', '0', '-z', '0.05', '-name', 'robotic_hand'],
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,           # Start Gazebo
        node_robot_state_publisher,  # Start publishing robot description
        spawn_robot              # Spawn the robotic hand in Gazebo
    ])
