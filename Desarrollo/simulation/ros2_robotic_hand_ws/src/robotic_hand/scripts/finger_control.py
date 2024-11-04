#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class FalangeControlNode(Node):
    def __init__(self):
        super().__init__('falange_control_node')
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('Falange Control Node iniciado.')

    def timer_callback(self):
        traj = JointTrajectory()
        traj.joint_names = ['joint_indice_0', 'joint_indice_1', 'joint_indice_2']
        point = JointTrajectoryPoint()
        point.positions = [0.5, 0.5, 0.5]  # Ajusta las posiciones deseadas
        point.time_from_start.sec = 2  # Tiempo para alcanzar la posición
        traj.points.append(point)
        self.publisher_.publish(traj)
        self.get_logger().info('Comando enviado al dedo índice.')

def main(args=None):
    rclpy.init(args=args)
    node = FalangeControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
