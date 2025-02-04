cmake_minimum_required(VERSION 3.5)
project(robotic_hand)

# Encontrar las dependencias necesarias
find_package(Python3 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(urdf REQUIRED)
find_package(gz-sim8 REQUIRED)  # Ajustado para Gazebo Sim Harmonic 8.6.0
find_package(trajectory_msgs REQUIRED)

# Instalar archivos URDF, mallas y archivos de lanzamiento
install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Instalar scripts de Python
install(PROGRAMS
  scripts/test_control01.py
  DESTINATION lib/${PROJECT_NAME}  # Aquí se instalan los scripts en el directorio lib
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY worlds/
    DESTINATION share/${PROJECT_NAME}/worlds
)

install(DIRECTORY description/
    DESTINATION share/${PROJECT_NAME}/description
)

ament_package()
