import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Replay Buffer for single-step transitions
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.ptr = 0

    def store(self, state, action, reward):
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, reward))
        else:
            self.buffer[self.ptr] = (state, action, reward)
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards = zip(*batch)
        return (torch.stack(states), torch.stack(actions), torch.tensor(rewards, dtype=torch.float32))

    def size(self):
        return len(self.buffer)


# Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, conv_channels=[16, 32, 64], hidden_size=256):
        super(ActorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(conv_channels[2] * 32 * 32, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions)

    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, conv_channels=[16, 32, 64], hidden_size=256):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(conv_channels[2] * 32 * 32 + n_actions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value


# TD3 Agent for single-step episodes
class TD3Agent:
    def __init__(self, input_dim, n_actions, actor_lr=0.001, critic_lr=0.001, gamma=0.99, tau=0.005, buffer_size=10000, batch_size=64, conv_channels=[16, 32, 64], hidden_size=256):
        self.actor = ActorNetwork(input_dim, n_actions, conv_channels, hidden_size)
        self.critic = CriticNetwork(input_dim, n_actions, conv_channels, hidden_size)
        self.target_actor = ActorNetwork(input_dim, n_actions, conv_channels, hidden_size)
        self.target_critic = CriticNetwork(input_dim, n_actions, conv_channels, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Initialize target networks with same weights as the original networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        action = self.actor(state).detach().numpy()[0]
        return action

    def store_transition(self, state, action, reward):
        self.replay_buffer.store(state, action, reward)

    def learn(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        states, actions, rewards = self.replay_buffer.sample(self.batch_size)

        # Critic loss
        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, rewards.unsqueeze(1))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_networks()

    def update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
