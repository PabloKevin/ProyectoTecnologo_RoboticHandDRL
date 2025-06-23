#!/usr/bin/env python3
import rclpy
import threading
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time
import requests
#import json
import numpy as np
#import os
#import cv2
from paho.mqtt import client as mqtt_client

class JointPublisher(Node):
    def __init__(self, hand):
        super().__init__('joint_publisher')
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        #self.timer = self.create_timer(0.1, self.publish_joint_states)  # Desactivado al usar hilos
        self.hand = hand
        # Guardamos la última posición para interpolar sin bloquear
        self.prev_positions = self.hand.joint_positions.copy()

    def publish_joint_states(self, ramp_steps=20, rate_hz=50):
        """
        Lanza un hilo que interpola de prev_positions → hand.joint_positions
        publicando pequeños pasos para un movimiento suave.
        """
        def _smooth(start, end):
            dt = 1.0 / rate_hz
            for k in range(1, ramp_steps + 1):
                alpha = k / float(ramp_steps)
                ramped = [s + alpha * (e - s) for s, e in zip(start, end)]
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.hand.joint_names
                msg.position = ramped
                self.publisher.publish(msg)
                time.sleep(dt)
            # Al terminar, fijamos la nueva posición como base
            self.prev_positions = end.copy()

        start = self.prev_positions.copy()
        end   = self.hand.joint_positions.copy()
        threading.Thread(target=_smooth, args=(start, end), daemon=True).start()

class Finger():
    def __init__(self, name, upper_lims, lower_lims=None, positions=None, position_max=2.0):
        self.name = name
        self.upper_lims = upper_lims
        if lower_lims is None:
            self.lower_lims = [0.0 for _ in range(len(upper_lims))]
        else:
            self.lower_lims = lower_lims
        self.joint_names = [f"{name}_joint{j}" for j in range(len(upper_lims))]
        if positions is None:
            self.positions = [0.0 for _ in range(len(upper_lims))]
        else:
            self.positions = positions
        self.position_max = position_max

    def action(self, position):
        positions = []
        for i in range(len(self.positions)):
            positions.append(
                (position * (self.upper_lims[i] - self.lower_lims[i]) / self.position_max)
                + self.lower_lims[i]
            )
        self.positions = positions
        return positions

class Hand():
    def __init__(self, pulgar=None, indice=None, medio=None, anular=None, menique=None):
        self.fingers = {
            "pulgar": pulgar,
            "indice": indice,
            "medio": medio,
            "anular": anular,
            "menique": menique
        }
        self.joint_names = [
            joint for finger in self.fingers.values() if finger is not None
            for joint in finger.joint_names
        ]

    @property
    def joint_positions(self):
        return [
            pos for finger in self.fingers.values() if finger is not None
            for pos in finger.positions
        ]

    def action(self, combination):
        for i, finger_name in enumerate(self.fingers):
            self.fingers[finger_name].action(combination[i])

class MQTT_publisher():
    def __init__(self, client_id, broker='localhost', port=1883):
        self.broker = broker
        self.port = port
        self.client_id = client_id

        self.client = self.connect_mqtt()
        self.client.loop_start()

    def connect_mqtt(self):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print("Failed to connect, return code %d\n", rc)

        client = mqtt_client.Client(self.client_id)
        client.on_connect = on_connect
        client.connect(self.broker, self.port)
        return client

    def publish(self, topic, msg):
        status = self.client.publish(topic, msg)[0]
        if status == 0:
            print(f"Send `{msg}` to topic `{topic}`")
        else:
            print(f"Failed to send message to topic {topic}")

    def stop_loop(self):
        self.client.loop_stop()

    def start_loop(self):
        self.client.loop_start()

def get_observation_img(img_of_interest, tool_name):
    img_params = {"img_of_interest": img_of_interest, "tool_name": tool_name}
    """Sends image params and returns received image."""
    API_URL = "http://127.0.0.1:8001/image"
    try:
        response = requests.post(API_URL, json=img_params)
        if response.status_code == 200:
            image = response.json()["image"]
            error = response.json()["error"]
            return image, error
        else:
            print(f"Error from server: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to agent: {e}")

def get_action(observation):
    """Sends observation to ML agent and returns received action."""
    API_URL = "http://127.0.0.1:8000/predict"
    try:
        response = requests.post(API_URL, json={"observation": observation})
        if response.status_code == 200:
            return response.json()["action"]
        else:
            print(f"Error from server: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to agent: {e}")

def main():
    rclpy.init()

    pulgar = Finger("pulgar", upper_lims=[-1.289, -0.533], lower_lims=[0.0, 0.395])
    indice = Finger("indice", upper_lims=[-1.281, -1.305, -0.880])
    medio = Finger("medio", upper_lims=[-1.322, -1.403, 0.953])
    anular = Finger("anular", upper_lims=[-1.305, -1.297, -1.065])
    menique = Finger("menique", upper_lims=[-1.117, -1.297, -1.094])

    left_hand = Hand(pulgar, indice, medio, anular, menique)
    node = JointPublisher(left_hand)

    # Spin de ROS en background para no bloquear input()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    agent_MQTT_pub = MQTT_publisher(client_id='agent_controller_publisher')
    time.sleep(1)  # Wait for MQTT connection

    try:
        while True:
            tool_in = input("Tool: ")
            tool_in = None if tool_in == "random" else tool_in
            image_received, error = get_observation_img("all", tool_in)
            if error:
                print(f"Herramienta no reconocida, error: {error}")
            action_received = get_action(image_received)
            combination = action_received["position"]
            print(f"Moving fingers to: -> {combination}")

            agent_MQTT_pub.publish(topic="RoboticHand_ML/action", msg=str(combination))
            left_hand.action(combination)
            node.publish_joint_states()

    except KeyboardInterrupt:
        print("\nStopping robot movement.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
