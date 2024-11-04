from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Rutas a archivos y directorios
    package_name = 'robotic_hand'
    sdf_path = PathJoinSubstitution(
        [get_package_share_directory(package_name), 'urdf', 'robotic_hand.sdf']
    )
    controller_config = PathJoinSubstitution(
        [get_package_share_directory(package_name), 'config', 'hand_controllers.yaml']
    )

    return LaunchDescription([
        # Iniciar Gazebo Sim Harmonic en un mundo vacío
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', 'empty.sdf'],  # Cambiado a 'gz' para Gazebo Sim
            output='screen',
        ),
        # Esperar a que Gazebo se inicie correctamente antes de continuar
        TimerAction(
            period=5.0,  # Esperar 5 segundos para asegurarse de que Gazebo esté completamente iniciado
            actions=[
                # Spawnear el robot usando el comando de Gazebo
                ExecuteProcess(
                    cmd=[
                        'gz', 'sim', '--spawn-file', sdf_path.perform(None),
                        '--name', 'robotic_hand', '--pose', '0 0 0 0 0 0'
                    ],
                    output='screen'
                )
            ]
        ),
        # Spawnear los controladores después de que el robot ha sido spawnado
        TimerAction(
            period=10.0,  # Esperar otros 5 segundos después de spawnear el robot
            actions=[
                # Inicializar controller_manager
                Node(
                    package='controller_manager',
                    executable='ros2_control_node',
                    parameters=[controller_config],
                    output='screen'
                ),
                # Spawner del joint_state_controller
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['joint_state_controller', '--controller-manager', '/controller_manager'],
                    output='screen',
                ),
                # Spawner del joint_trajectory_controller
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager'],
                    output='screen',
                ),
            ]
        ),
    ])
