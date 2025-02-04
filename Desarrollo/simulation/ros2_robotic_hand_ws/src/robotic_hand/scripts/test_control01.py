import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time

class JointPublisher(Node):
    def __init__(self, hand):
        super().__init__('joint_publisher')
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.1, self.publish_joint_states)  # Publish at 10Hz
        self.hand = hand

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.hand.joint_names
        msg.position = self.hand.joint_positions
        self.publisher.publish(msg)

        # Example: Move the first joint
        for action in [0.0, 1.0, 2.0]:
            self.hand.fingers["pulgar"].action(action)
            self.hand.fingers["indice"].action(action)
            time.sleep(1)


class Finger():
    def __init__(self, name, upper_lims, lower_lims=None, positions=None, position_max=2.0):
        self.name = name
        self.upper_lims = upper_lims
        if lower_lims is None:
            self.lower_lims=[0.0 for _ in range(len(upper_lims))]
        else:
            self.lower_lims=lower_lims
        self.joint_names = [f"{name}_joint{joint}" for joint in range(len(upper_lims))]
        if positions is None:
            self.positions = [0.0 for _ in range(len(upper_lims))]
        else:
            self.positions = positions
        self.position_max=position_max

    def action(self, position):
        positions = []
        for i in range(len(self.positions)):
            positions.append((position*(self.upper_lims[i]-self.lower_lims[i])/self.position_max)+self.lower_lims[i])
        self.positions=positions
        return positions
    
class Hand():
    def __init__(self, pulgar=None, indice=None, medio=None, anular=None, menique=None):
        self.fingers = {"pulgar": pulgar, "indice": indice, "medio": medio, "anular": anular, "menique": menique}
        self.joint_names = [joint for finger in self.fingers.values() if finger is not None for joint in finger.joint_names]
    
    @property
    def joint_positions(self):
        return [pos for finger in self.fingers.values() if finger is not None for pos in finger.positions]


        

def main():
    indice = Finger("indice", upper_lims=[-1.281, -1.305, -1.224])
    pulgar = Finger("pulgar", upper_lims=[-1.569, -0.806], lower_lims=[0.0, 0.395])

    left_hand = Hand(pulgar,indice)

    rclpy.init()
    node = JointPublisher(left_hand)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Catch manual Ctrl+C to avoid calling shutdown twice
    finally:
        node.destroy_node()
        if rclpy.ok():  # Only shutdown if ROS 2 is still active
            rclpy.shutdown()

if __name__ == '__main__':
    main()

