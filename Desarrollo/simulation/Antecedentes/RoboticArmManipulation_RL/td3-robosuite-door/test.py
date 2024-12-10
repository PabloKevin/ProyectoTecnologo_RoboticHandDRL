import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent


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

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.005,
                  input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0],
                  layer1_size=layer1_size, layer2_size=layer2_size, batch_size=batch_size)

    n_games = 3 
    best_score = 0
    episode_identifier = f"1 - actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} layer1_size={layer1_size} layer2_size={layer2_size}"


    agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation, validation=True) #Importante el validation=True para que no entre en warmup y haga acciones randoms
            next_observation, reward, done, info = env.step(action)
            env.render()
            score += reward
            observation = next_observation
            time.sleep(0.01)

        print(f"Episode: {i} Score: {score}")
