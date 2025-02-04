import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


from launch_ros.actions import Node
import xacro


def generate_launch_description():

    # Specify the name of the package and path to xacro file within the package
    pkg_name = 'robotic_hand'
    file_subpath = 'description/robotic_hand.urdf.xacro'


    # Use xacro to process the file
    xacro_file = os.path.join(get_package_share_directory(pkg_name),file_subpath)
    robot_description_raw = xacro.process_file(xacro_file).toxml()


    # Configure the node
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_raw,
        'use_sim_time': True}] # add other parameters here if required
    )



    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
        ))
    )

    spawn_entity = Node(package='ros_gz_sim', executable='spawn_entity.py',
                    arguments=['-topic', 'robot_description',
                                '-entity', 'my_bot'],
                    output='screen')

    
    spawn_robot_cmd = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-string', robot_description_raw, '-x', '0', '-y', '0', '-z', '0.05', '-name', 'robotic_hand']
    )





    # Run the node
    return LaunchDescription([
        node_robot_state_publisher,
        gazebo,
        spawn_entity,
        #spawn_robot_cmd
    ])