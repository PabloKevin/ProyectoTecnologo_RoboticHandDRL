# Replay Buffer for single-step transitions

import numpy as np

# (state, action, reward, next_state, done)  estas features/parámetros son comunes en muchos algoritmos de RL. 
# Dado que este el problema consta de simple-step episodes, no se necesitan next_state y done.
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, n_choices_per_finger=3):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float64)  # Binary image: dtype=np.uint8.  *input_shape the "*" unpack the elements in the iterable as different arguments
        self.action_memory = np.zeros((self.mem_size, n_actions, n_choices_per_finger), dtype=np.float64)  # Discrete actions: dtype=np.uint8
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float64)  # Rewards remain float32

    def store_transition(self, state, action, reward):
        index = self.mem_cntr % self.mem_size  # Circular buffer logic
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]

        return states, actions, rewards

