import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

# Obtener la ruta de instalación del paquete
packagepath = get_package_share_directory('robotic_hand')  # Cambiar 'nav_car' por 'robotic_hand'
print(packagepath)
print(os.path.dirname(__file__))

carmodel = packagepath + '/urdf/robotic_hand.sdf'  # Actualizar la ruta al modelo
robot_desc = open(carmodel).read()

def generate_launch_description():
    gazebo_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch',
            'gz_sim.launch.py')),
        launch_arguments={'gz_args': packagepath + '/worlds/empty.sdf -r -v4'}.items()  # Cambiar a tu archivo de mundo
    )

    robot_to_gazebo = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-string', robot_desc, '-x', '0', '-y', '0', '-z', '0.05', '-name', 'robotic_hand']
    )

    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': packagepath + '/config/hand_controllers.yaml'}]  # Cambiar a la ruta correcta del archivo de configuración
    )

    robot_desc_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': robot_desc},
        ]
    )

    return LaunchDescription([
        gazebo_node,
        robot_to_gazebo,
        bridge_node,
        robot_desc_node,
    ])
