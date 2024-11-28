#18.16
import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, actor_learning_rate, critic_learning_rate, input_dims, tau, env, gamma=0.99, update_actor_interval=2, warmup=1000, 
                 n_actions=2, max_size=1000000, layer1_size=256, layer2_size=128, batch_size=100, noise=0.1):

        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        # Create the networks
        self.actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size, 
                                  fc2_dims=layer2_size, n_actions=n_actions, name='actor', 
                                  learning_rate=actor_learning_rate)

        self.critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, 
                                      fc2_dims=layer2_size, n_actions=n_actions, 
                                      name='critic_1', learning_rate=critic_learning_rate)

        self.critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, 
                                      fc2_dims=layer2_size, n_actions=n_actions, 
                                      name='critic_2', learning_rate=critic_learning_rate)

        # Create the target networks
        self.target_actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size, 
                                         fc2_dims=layer2_size, n_actions=n_actions, 
                                         name='target_actor', learning_rate=actor_learning_rate)

        self.target_critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, 
                                             fc2_dims=layer2_size, n_actions=n_actions, 
                                             name='target_critic_1', learning_rate=critic_learning_rate)

        self.target_critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                             fc2_dims=layer2_size, n_actions=n_actions,
                                             name='target_critic_2', learning_rate=critic_learning_rate)

        self.noise = noise
        self.update_network_parameters(tau=1)


    def choose_action(self, observation, validation=False):
        # Los primeros episodios (de warm up) son con acciones randoms para permitir al agente explorar el entorno y sus reacciones,
        # formar una política. validation = True, permite saltar el warmup (que es un int), podría ser si ya tiene entrenamiento.

        if self.time_step < self.warmup and validation is False:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)

        # Luego del warmup, las acciones se desarrollan en función del estado, con una política ya creada (que va a seguir mejorando).
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)    #el .to(self.actor.device) manda el mu a la GPU o CPU (según corresponda)

        # Para mejorar el entrenamiento, se le añade un ruido normal a la acción, para ayudar a explotar durante todo el entrenamiento.
        # Esto es útil sobre todo para espacios de acciones continuos (no discretos) o muy grandes, en lo que además se enfoca el algoritmo td3.
        # Pero el fundamento de no cerrarse en la política es interesante para espacios de acciones discretos, probarlo en el problema.
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)

        # Más allá del espacio continuo, pueden haber restricciones físicas u otras, entonces el .clamp se asegura que el ruido no marque acciones
        # fuera de los límites de acción. Ej: que la velocidad de motores no sea mayor a 50.
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])

        self.time_step += 1

        # manda el mu_prime a la cpu, lo transforma en un tensor, y luego lo transforma en un valor np para leer la acción y ejecutarla o mandarla a otro lado.
        return mu_prime.cpu().detach().numpy()  
