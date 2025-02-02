import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time

class JointPublisher(Node):
    def __init__(self):
        super().__init__('joint_publisher')
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.1, self.publish_joint_states)  # Publish at 10Hz
        self.joint_positions = [0.0, 0.0, 0.0]  # Initial positions for three joints

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["joint_indice_0", "joint_indice_1", "joint_indice_2"]  # Change to your joint names
        msg.position = self.joint_positions
        self.publisher.publish(msg)

        # Example: Move the first joint
        self.joint_positions[0] += 0.05
        if self.joint_positions[0] > 1.5:
            self.joint_positions[0] = -1.5

def main():
    rclpy.init()
    node = JointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
