import time
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from td3_torch import Agent
from custom_hand_env import ToolManipulationEnv


if __name__ == '__main__':

    if not os.path.exists("Desarrollo/simulation/Env03/tmp/td3"):
        os.makedirs("Desarrollo/simulation/Env03/tmp/td3")

    # Create an instance of your custom environment
    env = ToolManipulationEnv(image_shape=(256, 256, 1), n_fingers=5)
# Ir probando con numeros más simples para que lleve menos tiempo. Dado que el problemas es más simple,
# usar menos neuronas, probablemente no necesite tantas imágenes para aprender. Quizá probar con 1 sola capa.

    actor_learning_rate = 0.001 #0.001
    critic_learning_rate = 0.001 #0.001
    batch_size = 128 #128

    conv_channels=[8, 16, 32] #[16, 32, 64]
    hidden_size=128 #256
    warmup = 1000
    
    env.reward_weights["reward_alpha"] = 1

    # Reduce the replay buffer size
    max_size = 10000  # Adjust this value based on your memory capacity

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.005,
                  input_dims=env.get_observation_space_shape(), env=env, n_actions=env.n_fingers, 
                  n_choices_per_finger=env.n_choices_per_finger, conv_channels=conv_channels, hidden_size=hidden_size, 
                  batch_size=batch_size, warmup=warmup, max_size=max_size) 

    agent.load_models()

    episodes = 3 #10000

    scores = []
    for i in range(episodes):
        observation = env.reset()
        score = 0
        #action_probs = np.array(env.probabilities_of_interest)[np.random.randint(0, len(env.probabilities_of_interest))]
        action_probs = agent.choose_action(observation, validation=True)
        action = agent.env.probs2actions(action_probs)
        next_observation, reward, info = env.step(action)
        env.render(timeout=None)
        score += reward

        scores.append(score)

        #print(f"Episode: {i}; Score: {score}; Action: {action}")

    print(f"Test episodes: {episodes}")
    print(f"Average score: {np.mean(scores)}")
    print(f"Accuracy: {scores.count(5.0)/len(scores)}")