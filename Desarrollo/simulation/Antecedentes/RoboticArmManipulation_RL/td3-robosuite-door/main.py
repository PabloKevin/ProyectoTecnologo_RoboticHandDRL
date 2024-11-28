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

    # https://robosuite.ai/docs/modules/environments.html
    env = suite.make(
        env_name,                   
        robots=["Panda"],           # load a Panda robot
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),       # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
        has_renderer = False,       # on-screen rendering
        use_camera_obs = False,     # no observations needed
        horizon = 300,              # each episode terminates after 300 steps. Ajustar según problema.
        reward_shaping=True,        # use a dense reward signal for learning
        control_freq=20,            # 20 hz control for applied actions
    )

    env = GymWrapper(env) #pone el env en el formato esperado para gym, se podría usar un env fuera de robosuite


    ###
    critic_network = CriticNetwork(input_dims=[8], n_actions=8)
    actor_network = ActorNetwork(input_dims=[8], fc1_dims=8)

    replay_buffer = ReplayBuffer(max_size=8, input_shape=[8], n_actions=8)