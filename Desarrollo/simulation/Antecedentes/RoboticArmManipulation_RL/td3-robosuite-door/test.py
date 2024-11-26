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
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer = True,
        use_camera_obs = False,
        horizon = 300,
        render_camera = "frontview",
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)

