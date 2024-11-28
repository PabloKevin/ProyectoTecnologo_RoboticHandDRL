import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper


if __name__ == '__main__':

    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")

    env_name = "Door"

    env = suite.make(
        env_name,
        robots=["Panda"],                   # load a Panda robot
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
        has_renderer = True,                # on-screen rendering
        use_camera_obs = False,             # no observations needed
        horizon = 300,                      # each episode terminates after 300 steps. Ajustar según problema.
        render_camera = "frontview",        # visualize the "frontview" camera
        has_offscreen_renderer=True,        # no off-screen rendering
        reward_shaping=True,                # use a dense reward signal for learning
        control_freq=20,                    # 20 hz control for applied actions
    )

    env = GymWrapper(env)       #pone el env en el formato esperado para gym, se podría usar un env fuera de robosuite

