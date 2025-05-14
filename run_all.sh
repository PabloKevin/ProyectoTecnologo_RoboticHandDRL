#!/usr/bin/env bash

gnome-terminal --tab --title="ROS RViz" -- bash -ic "
  echo '=== ROS RViz Terminal ===';
  cd Desarrollo/simulation/ros2_robotic_hand_ws/;
  source install/setup.bash;
  ros2 launch robotic_hand ros_rviz.launch.py;
  exec bash
"

gnome-terminal --tab --title="Image Generator" -- bash -ic "
  echo '=== Image Generator Terminal ===';
  cd Desarrollo/simulation/Env04/;
  conda activate RoboticHand_ML;
  python3 Image_generator_server.py;
  exec bash
"

gnome-terminal --tab --title="Agent Server" -- bash -ic "
  echo '=== Agent Server Terminal ===';
  cd Desarrollo/simulation/Env04/;
  conda activate RoboticHand_ML;
  python3 Agent_server.py;
  exec bash
"

gnome-terminal --tab --title="Agent Controller" -- bash -ic "
  echo '=== Agent Controller Terminal ===';
  cd Desarrollo/simulation/ros2_robotic_hand_ws/;
  source install/setup.bash;
  conda deactivate;
  echo 'Waiting 5 seconds for other processes...';
  sleep 10;  # <-- Wait 10 seconds here;
  ros2 run robotic_hand agent_controller.py;
  echo 'last command: ros2 run robotic_hand agent_controller.py';
  exec bash
"
