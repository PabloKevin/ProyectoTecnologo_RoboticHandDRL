import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ObserverNetwork
from torch.optim.lr_scheduler import StepLR

class Agent:
    def __init__(self, env, actor_learning_rate=0.003, critic_learning_rate=0.0008, tau=0.005, gamma=0.99, update_actor_interval=2, warmup=1000, 
                 max_size=1000000, hidden_layers=[64,32], batch_size=64, noise=0.1, checkpoint_dir='Desarrollo/simulation/Env04/tmp/td3'):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learn_step_cntr = 0 
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = env.n_fingers
        self.update_actor_iter = update_actor_interval
        self.env = env
        self.min_action = -1.0 # Para ser concistente con la red, después se cambia a [0,2]
        self.max_action = 1.0

        # Create and load observer
        #self.observer = ObserverNetwork()
        #self.observer.load_model()
        
        #self.input_dims = self.observer.output_dims + 1 # +1 = f_idx (finger index)
        self.input_dims = 11 # observer_output (1) + f_idx (1)
        
        self.memory = ReplayBuffer(max_size, self.input_dims, self.n_actions)

        self.checkpoint_dir = checkpoint_dir 

        # Create the networks
        self.actor = ActorNetwork(input_dims=self.input_dims, hidden_layers=hidden_layers, n_actions=self.n_actions,
                                  name='actor', learning_rate=actor_learning_rate, checkpoint_dir=self.checkpoint_dir)

        self.critic_1 = CriticNetwork(input_dims = self.input_dims + self.n_actions, hidden_layers=hidden_layers,
                                      name='critic_1', learning_rate=critic_learning_rate, checkpoint_dir=self.checkpoint_dir)

        self.critic_2 = CriticNetwork(input_dims = self.input_dims + self.n_actions, hidden_layers=hidden_layers, 
                                      name='critic_2', learning_rate=critic_learning_rate, checkpoint_dir=self.checkpoint_dir)
        
        self.actor_scheduler = StepLR(self.actor.optimizer,
                                      step_size=4000,   # cada 1500 episodios
                                      gamma=0.8)       # factor de decaimiento
        self.critic1_scheduler = StepLR(self.critic_1.optimizer,
                                       step_size=4000,
                                       gamma=0.8)
        self.critic2_scheduler = StepLR(self.critic_2.optimizer,
                                       step_size=4000,
                                       gamma=0.8)

        # Create the target networks
        self.target_actor = ActorNetwork(input_dims=self.input_dims, hidden_layers=hidden_layers, n_actions=self.n_actions, 
                                         name='target_actor', learning_rate=actor_learning_rate, checkpoint_dir=self.checkpoint_dir)

        self.target_critic_1 = CriticNetwork(input_dims = self.input_dims + self.n_actions, hidden_layers=hidden_layers,
                                             name='target_critic_1', learning_rate=critic_learning_rate, checkpoint_dir=self.checkpoint_dir)

        self.target_critic_2 = CriticNetwork(input_dims = self.input_dims + self.n_actions, hidden_layers=hidden_layers,
                                             name='target_critic_2', learning_rate=critic_learning_rate, checkpoint_dir=self.checkpoint_dir)
        
        self.target_actor_scheduler = StepLR(self.target_actor.optimizer,
                                      step_size=4000,   # cada 1500 episodios
                                      gamma=0.8)       # factor de decaimiento
        self.target_critic1_scheduler = StepLR(self.target_critic_1.optimizer,
                                       step_size=4000,
                                       gamma=0.8)
        self.target_critic2_scheduler = StepLR(self.target_critic_2.optimizer,
                                       step_size=4000,
                                       gamma=0.8)
        
        """# Initialize weights for all networks
        def initialize_weights(m):
            if isinstance(m, T.nn.Linear) or isinstance(m, T.nn.Conv2d):
                T.nn.init.kaiming_uniform_(m.weight,  nonlinearity='leaky_relu')
                if m.bias is not None:
                    T.nn.init.zeros_(m.bias)

        self.actor.apply(initialize_weights)
        self.critic_1.apply(initialize_weights)
        self.critic_2.apply(initialize_weights)
        self.target_actor.apply(initialize_weights)
        self.target_critic_1.apply(initialize_weights)
        self.target_critic_2.apply(initialize_weights)"""
        

        self.noise = noise
        self.update_networks_parameters(tau=1)
        """self.epsilon = 0.4
        self.epsilon_decay = 0.8
        self.min_epsilon = 0.1 #0.1"""
        self.comb_interest = 0
        self.idx = 0
        self.idx02 = 1

    def choose_action(self, observation, validation=False):
        # Los primeros episodios (de warm up) son con acciones randoms para permitir al agente explorar el entorno y sus reacciones,
        # formar una política. validation = True, permite saltar el warmup, podría ser si ya tiene entrenamiento.
        
        if self.time_step < self.warmup and validation is False:
            mu = T.tensor(np.random.uniform(low=-1, high=1, size=(self.n_actions,))).to(self.actor.device)

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
        mu_prime = T.clamp(mu_prime, min=self.min_action, max=self.max_action)

        self.time_step += 1

        # manda el mu_prime a la cpu, lo transforma en un tensor, y luego lo transforma en un valor np para leer la acción y ejecutarla o mandarla a otro lado.
        return mu_prime.cpu().detach().numpy() 

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        # Asegura suficientes transiciones en memoria antes de entrenar, evitando aprendizaje prematuro y fomentando una exploración inicial.
        # Evitar un aprendizaje prematuro: Si el agente empieza a entrenar demasiado pronto, puede intentar ajustar las redes con datos insuficientes o poco representativos 
        # del entorno, lo cual podría resultar en un entrenamiento inestable o ineficaz.
        # Reforzar exploración inicial: Durante las primeras iteraciones, el agente debería enfocarse más en explorar el entorno para recopilar información útil. 
        if self.memory.mem_ctr < self.batch_size * 10: 
            return

        # Obtiene un batch de muestras aleatorias desde el Replay Buffer
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        # Convierte las muestras del batch a tensores y las mueve al dispositivo adecuado (CPU o GPU).
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        # Genera las acciones objetivo utilizando la red objetivo del actor,
        # y agrega ruido para evitar sobreestimaciones de los valores Q.
        target_actions = self.target_actor.forward(next_state)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        # Asegura que las acciones objetivo estén dentro de los límites permitidos.
        target_actions = T.clamp(target_actions, self.min_action, self.max_action)

        # Calcula los valores Q objetivo utilizando las redes críticas objetivo y las acciones objetivo.
        next_q1 = self.target_critic_1.forward(next_state, target_actions)
        next_q2 = self.target_critic_2.forward(next_state, target_actions)
        
        # Calcula los valores Q actuales usando las redes críticas principales.
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        next_q1[done] = 0.0
        next_q2[done] = 0.0

        next_q1 = next_q1.view(-1)
        next_q2 = next_q2.view(-1)

        # Utiliza el valor Q mínimo para reducir el riesgo de sobreestimación.
        next_critic_value = T.min(next_q1, next_q2)

        # Además de la recompensa del estado actual, da buena (casi igual) importancia a la recompensa del estado siguiente (futuro), lo que ayuda en el aprendizaje
        # porque un estado "malo" puede en realidad ser el camino a un estado siguiente muy bueno. Revisar ecuación de Bellman. 
        # El siguiente approach también es interesante: q_target = reward + self.gamma * next_q * (1 - done) .
        target = reward + self.gamma * next_critic_value
        target = target.view(self.batch_size, 1)

        # Reinicia los gradientes antes de realizar las actualizaciones.
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        # Calcula las pérdidas de las redes críticas usando Mean Squared Error (MSE) entre los valores Q actuales y los objetivos.
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)

        # Retropropaga las pérdidas.
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        # Actualiza los parámetros de las redes críticas.
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Incrementa el contador de pasos de aprendizaje.
        self.learn_step_cntr += 1

        # Actualiza la red del actor solo después de un número determinado de pasos.
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        # Calcula la pérdida del actor basada en las predicciones de la critic_1, tratando de maximizar los valores Q de las acciones generadas.
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state)) 
        # Durante el entrenamiento, se trabaja con batches de datos, con lo cual el actor_q1_loss tendrá el Q_value para cada estado y acción del batch, 
        # y por eso tiene sentido el .mean() (si en vez de batch, fuera un solo Q_value, no tendría sentido hacer el promedio), para no darle sobreimportancia al tamaño del batch.
        actor_loss = -T.mean(actor_q1_loss) #Como los optimizadores típicamente minimizan una función de pérdida, negamos el valor Q para convertir la tarea en un problema de minimización.
        
        # Retropropaga la pérdida y actualiza los parámetros de la red del actor.
        actor_loss.backward()
        self.actor.optimizer.step()

        # Realiza una actualización suave (soft update) de las redes objetivo.
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

    def train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.target_actor.train()
        self.target_critic_1.train()
        self.target_critic_2.train()