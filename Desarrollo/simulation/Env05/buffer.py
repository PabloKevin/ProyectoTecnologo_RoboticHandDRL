# Replay Buffer for single-step transitions

import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_dims, n_actions, seq_len=5):
        self.mem_size = max_size
        self.seq_len = seq_len
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_ctr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done
        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        valid_indices = np.arange(max_mem - self.seq_len)
        batch_starts = np.random.choice(valid_indices, batch_size, replace=False)

        states = []
        actions = []
        rewards = []
        new_states = []
        dones = []

        for start in batch_starts:
            end = start + self.seq_len
            states.append(self.state_memory[start:end])
            actions.append(self.action_memory[start:end])
            rewards.append(self.reward_memory[start:end])
            new_states.append(self.new_state_memory[start:end])
            dones.append(self.terminal_memory[start:end])

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(new_states), np.array(dones))

