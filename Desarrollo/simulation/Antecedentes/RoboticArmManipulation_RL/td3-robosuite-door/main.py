import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper

from networks import CriticNetwork,ActorNetwork
from buffer import ReplayBuffer

if __name__ == '__main__':

    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")

    env_name = "Door"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer = False,
        use_camera_obs = False,
        horizon = 300,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)


    ###
    critic_network = CriticNetwork(input_dims=[8], n_actions=8)
    actor_network = ActorNetwork(input_dims=[8], fc1_dims=8)

    replay_buffer = ReplayBuffer(max_size=8, input_shape=[8], n_actions=8)