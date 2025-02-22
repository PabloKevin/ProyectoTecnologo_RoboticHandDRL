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
    rviz_config_file = os.path.join(get_package_share_directory(pkg_name), 'config', 'robotic_hand_v1.rviz')

    # Process URDF
    xacro_file = os.path.join(get_package_share_directory(pkg_name), file_subpath)
    robot_description_raw = xacro.process_file(xacro_file).toxml()

    # Robot State Publisher (publishes TF)
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_raw,
                     'use_sim_time': True}]
    )

    # RViz2 for visualization
    node_rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    return LaunchDescription([
        node_robot_state_publisher,
        node_rviz2
    ])
