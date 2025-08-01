import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, actor_learning_rate, critic_learning_rate, input_dims, tau, env, gamma=0.99, update_actor_interval=2, warmup=1000, 
                 n_actions=5, n_choices_per_finger=3, max_size=1000000, conv_channels=[16, 32, 64], hidden_size=256, batch_size=100, noise=0.1):

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0 
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.n_choices_per_finger = n_choices_per_finger
        self.update_actor_iter = update_actor_interval
        self.env = env

        # Create the networks
        self.actor = ActorNetwork(input_dims=input_dims, conv_channels=conv_channels, 
                                  hidden_size=hidden_size , n_actions=n_actions, n_choices_per_finger=n_choices_per_finger, name='actor', 
                                  learning_rate=actor_learning_rate)

        self.critic_1 = CriticNetwork(input_dims=input_dims, conv_channels=conv_channels, 
                                      hidden_size=hidden_size, n_actions=n_actions, 
                                      name='critic_1', learning_rate=critic_learning_rate)

        self.critic_2 = CriticNetwork(input_dims=input_dims, conv_channels=conv_channels, 
                                      hidden_size=hidden_size, n_actions=n_actions, 
                                      name='critic_2', learning_rate=critic_learning_rate)

        # Create the target networks
        self.target_actor = ActorNetwork(input_dims=input_dims, conv_channels=conv_channels, 
                                         hidden_size=hidden_size, n_actions=n_actions, n_choices_per_finger=n_choices_per_finger,
                                         name='target_actor', learning_rate=actor_learning_rate)

        self.target_critic_1 = CriticNetwork(input_dims=input_dims, conv_channels=conv_channels, 
                                             hidden_size=hidden_size, n_actions=n_actions, 
                                             name='target_critic_1', learning_rate=critic_learning_rate)

        self.target_critic_2 = CriticNetwork(input_dims=input_dims, conv_channels=conv_channels, 
                                             hidden_size=hidden_size, n_actions=n_actions,
                                             name='target_critic_2', learning_rate=critic_learning_rate)
        
        # Initialize weights for all networks
        def initialize_weights(m):
            if isinstance(m, T.nn.Linear) or isinstance(m, T.nn.Conv2d):
                T.nn.init.kaiming_uniform_(m.weight,  nonlinearity='relu')
                if m.bias is not None:
                    T.nn.init.zeros_(m.bias)

        self.actor.apply(initialize_weights)
        self.critic_1.apply(initialize_weights)
        self.critic_2.apply(initialize_weights)
        self.target_actor.apply(initialize_weights)
        self.target_critic_1.apply(initialize_weights)
        self.target_critic_2.apply(initialize_weights)

        self.noise = noise
        self.update_networks_parameters(tau=1)
        self.epsilon = 0.4
        self.epsilon_decay = 0.8
        self.min_epsilon = 0.1

    def choose_action(self, observation, validation=False):
        # Los primeros episodios (de warm up) son con acciones randoms para permitir al agente explorar el entorno y sus reacciones,
        # formar una política. validation = True, permite saltar el warmup (que es un int), podría ser si ya tiene entrenamiento.
        
        if self.time_step < self.warmup and validation is False:
            # Asegurar que cada cierto tiempo se ejecute una de las acciones de interés
            if self.time_step % 4 == 0:
                action_probs = T.tensor(np.array(self.env.probabilities_of_interest)[np.random.randint(0, len(self.env.probabilities_of_interest))], dtype=T.float).to(self.actor.device)
            # Use tensor to generate random actions
            else:
                random_probs = np.random.dirichlet(np.ones(self.n_choices_per_finger), size=self.n_actions)
                action_probs = T.tensor(random_probs, dtype=T.float).to(self.actor.device)
        else:
            # Asegurar que cada cierto tiempo (cada vez menos) se ejecute una de las acciones de interés
            if np.random.random() < self.epsilon and validation is False:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
                action_probs = T.tensor(np.array(self.env.probabilities_of_interest)[np.random.randint(0, len(self.env.probabilities_of_interest))], dtype=T.float).to(self.actor.device)
            # Use tensor to generate random actions
            else:
                # permute(0, 3, 1, 2) rearranges the dimensions from [height, width, channels] to [channels, height, width],
                # which is the format expected by PyTorch convolutional layers.
                state = T.tensor(observation, dtype=T.float).permute(2, 0, 1).to(self.actor.device) #tiene que ser float para que no haya problemas con la multiplicación de los pesos de la red
                action_probs = self.actor.forward(state).to(self.actor.device)

        self.time_step += 1

        return action_probs.cpu().detach().numpy() # Convert action to NumPy array and return

    def remember(self, state, action, reward):
        self.memory.store_transition(state, action, reward)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size * 10:
            return

        # Sample a batch from the replay buffer
        state, action, reward = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).permute(0, 3, 1, 2).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)

        # Update Critic networks
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        # Compute target Q-value using the target networks
        with T.no_grad():
            # For single-step episodes, the target is the immediate reward
            target_q1 = self.target_critic_1.forward(state, action)
            target_q2 = self.target_critic_2.forward(state, action)
            target_q = T.min(target_q1, target_q2)  # TD3 minimizes Q-value to reduce overestimation
            target = reward.view(-1, 1) + 0.0 * target_q  # 0.0 * target_q ensures compatibility with TD3 logic

        # Compute current Q-values from the Critic networks
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        # Compute Critic loss
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        # Clip Critic gradients
        T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=5.0)
        T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=5.0)

        noise_scale = 1e-5

        for param in self.critic_1.parameters():
            if param.grad is not None:
                param.grad += T.randn_like(param.grad) * noise_scale

        # Update Critic optimizers
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        # Update Actor network at specified intervals
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()

        # Compute Actor loss
        actor_probabilities = self.actor.forward(state)
        actor_q1_loss = self.critic_1.forward(state, actor_probabilities)
        actor_loss = -T.mean(actor_q1_loss)  # Maximize Q-value for the Actor
        actor_loss.backward()

        # Clip Actor gradients
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)

        noise_scale = 1e-3

        for param in self.actor.parameters():
            if param.grad is not None:
                param.grad += T.randn_like(param.grad) * noise_scale

        # Update Actor optimizer
        self.actor.optimizer.step()

        # Debugging: Log gradient norms for the Actor and Critic networks
        """print("Actor")
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                print(f"Layer {name} gradient norm: {param.grad.norm().item():.10f}")
            else:
                print(f"Layer {name} has no gradient")

        print("\nCritic 1")
        for name, param in self.critic_1.named_parameters():
            if param.grad is not None:
                print(f"Layer {name} gradient norm: {param.grad.norm().item():.10f}")
            else:
                print(f"Layer {name} has no gradient")"""

        # Update target networks
        self.update_networks_parameters()


    def update_networks_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
        print("Successfully loaded models")
