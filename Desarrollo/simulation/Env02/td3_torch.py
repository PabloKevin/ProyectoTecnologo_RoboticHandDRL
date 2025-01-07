"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) es un algoritmo de Deep Reinforcement Learning diseñado para trabajar con espacios de acción continuos. 
Es una mejora sobre el algoritmo DDPG (Deep Deterministic Policy Gradient) y aborda sus principales debilidades, como la sobreestimación de los valores Q. 
TD3 utiliza tres estrategias principales para mejorar la estabilidad y la precisión del aprendizaje:

* Principales características de TD3:
- Uso de dos redes críticas (Twin Critics):
En lugar de usar una sola red para estimar Q(s,a), TD3 utiliza dos redes críticas independientes.
Durante el entrenamiento, utiliza el valor más bajo de las dos estimaciones (min(Q1,Q2)) para reducir la sobreestimación del valor Q. Esto hace que las actualizaciones 
sean más conservadoras y evita políticas subóptimas.

- Política retardada (Delayed Policy Updates):
El actor (red de políticas) no se actualiza con cada paso de tiempo. En su lugar, se actualiza después de varios pasos de entrenamiento de las redes críticas.
Esto permite que las redes críticas converjan mejor antes de actualizar la política, lo que estabiliza el aprendizaje.

- Ruido dirigido a la acción (Target Policy Smoothing):
Se agrega ruido gaussiano a las acciones objetivo (a') antes de calcular los valores Q. Esto actúa como un mecanismo de regularización que ayuda a manejar el error de 
extrapolación y evita que el modelo confíe en estimaciones extremas de Q.

* Flujo general del algoritmo:
1. Se utilizan las redes críticas para calcular el valor esperado Q(s,a) basado en las transiciones almacenadas en el Replay Buffer.
2. El actor se entrena para maximizar el valor Q(s,a), es decir, para elegir acciones que conduzcan a mayores recompensas.
3. Las redes objetivo (actor y críticos) se actualizan gradualmente utilizando un "soft update" para garantizar estabilidad.
"""

import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent:
    # Va a estar compuesto en total por 6 redes neuronales
    def __init__(self, actor_learning_rate, critic_learning_rate, input_dims, tau, env, gamma=0.99, update_actor_interval=2, warmup=1000, 
                 n_actions=5, max_size=1000000, conv_channels=[16, 32, 64], hidden_size=256, batch_size=100, noise=0.1):

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.env = env

        # Create the networks
        self.actor = ActorNetwork(input_dims=input_dims, conv_channels=[16, 32, 64], 
                                  hidden_size=256 , n_actions=n_actions, name='actor', 
                                  learning_rate=actor_learning_rate)

        self.critic_1 = CriticNetwork(input_dims=input_dims, conv_channels=[16, 32, 64], 
                                      hidden_size=256, n_actions=n_actions, 
                                      name='critic_1', learning_rate=critic_learning_rate)

        self.critic_2 = CriticNetwork(input_dims=input_dims, conv_channels=[16, 32, 64], 
                                      hidden_size=256, n_actions=n_actions, 
                                      name='critic_2', learning_rate=critic_learning_rate)

        # Create the target networks
        self.target_actor = ActorNetwork(input_dims=input_dims, conv_channels=[16, 32, 64], 
                                         hidden_size=256, n_actions=n_actions, 
                                         name='target_actor', learning_rate=actor_learning_rate)

        self.target_critic_1 = CriticNetwork(input_dims=input_dims, conv_channels=[16, 32, 64], 
                                             hidden_size=256, n_actions=n_actions, 
                                             name='target_critic_1', learning_rate=critic_learning_rate)

        self.target_critic_2 = CriticNetwork(input_dims=input_dims, conv_channels=[16, 32, 64], 
                                             hidden_size=256, n_actions=n_actions,
                                             name='target_critic_2', learning_rate=critic_learning_rate)

        self.noise = noise
        self.update_networks_parameters(tau=1)


    def choose_action(self, observation, validation=False):
        # Los primeros episodios (de warm up) son con acciones randoms para permitir al agente explorar el entorno y sus reacciones,
        # formar una política. validation = True, permite saltar el warmup (que es un int), podría ser si ya tiene entrenamiento.

        if self.time_step < self.warmup and validation is False:
            # Asegurar que cada cierto tiempo se ejecute una de las acciones de interés
            if self.time_step % 2 == 0:
                action = T.tensor(np.array(self.env.combinations_of_interest)[np.random.randint(0, len(self.env.combinations_of_interest))], dtype=T.uint8).to(self.actor.device)
            # Use tensor to generate random actions
            else:
                action = T.tensor(self.env.action_space.sample(), dtype=T.uint8).to(self.actor.device)
        else:
            # Asegurar que cada cierto tiempo (cada vez menos) se ejecute una de las acciones de interés
            if self.time_step % int(np.sqrt(self.time_step)-20.0) == 0 and validation is False:
                action = T.tensor(np.array(self.env.combinations_of_interest)[np.random.randint(0, len(self.env.combinations_of_interest))], dtype=T.uint8).to(self.actor.device)
            # Use tensor to generate random actions
            else:
                state = T.tensor(observation, dtype=T.float).to(self.actor.device) #check the dtypes
                action = self.actor.forward(state).to(self.actor.device)

        # Convert action to NumPy array
        action = action.cpu().detach().numpy()

        self.time_step += 1
        return action

    def remember(self, state, action, reward):
        self.memory.store_transition(state, action, reward)

    def learn(self):
        # Asegura suficientes transiciones en memoria antes de entrenar, evitando aprendizaje prematuro y fomentando una exploración inicial.
        # Evitar un aprendizaje prematuro: Si el agente empieza a entrenar demasiado pronto, puede intentar ajustar las redes con datos insuficientes o poco representativos 
        # del entorno, lo cual podría resultar en un entrenamiento inestable o ineficaz.
        # Reforzar exploración inicial: Durante las primeras iteraciones, el agente debería enfocarse más en explorar el entorno para recopilar información útil. 
        if self.memory.mem_cntr < self.batch_size * 10: 
            return

        # Obtiene un batch de muestras aleatorias desde el Replay Buffer
        state, action, reward = self.memory.sample_buffer(self.batch_size)

        # Convierte las muestras del batch a tensores y las mueve al dispositivo adecuado (CPU o GPU).
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        
        
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        # Genera las acciones objetivo utilizando la red objetivo del actor,
        # y agrega ruido para evitar sobreestimaciones de los valores Q.
        target_actions = self.actor.forward(state)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        # Asegura que las acciones objetivo estén dentro de los límites permitidos.
        target_actions = T.clamp(target_actions, 0, 1)

        # Calcula los valores Q objetivo utilizando las redes críticas objetivo y las acciones objetivo.
        next_q1 = self.critic_1.forward(state, target_actions)
        next_q2 = self.critic_2.forward(state, target_actions)
        
        # Calcula los valores Q actuales usando las redes críticas principales.
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        
        

        next_q1 = next_q1.view(-1)
        next_q2 = next_q2.view(-1)

        # Utiliza el valor Q mínimo para reducir el riesgo de sobreestimación.
        next_critic_value = T.min(next_q1, next_q2)

        # Además de la recompensa del estado actual, da buena (casi igual) importancia a la recompensa del estado siguiente (futuro), lo que ayuda en el aprendizaje
        # porque un estado "malo" puede en realidad ser el camino a un estado siguiente muy bueno. Revisar ecuación de Bellman. 
        # El siguiente approach también es interesante: q_target = reward + self.gamma * next_q * (1 - done) .
        target = reward
        target = target.view(-1, 1)

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
        # Si no se proporciona un valor para tau, se utiliza el valor por defecto definido en la clase.
        if tau is None:
            tau = self.tau

        # Recupera los parámetros actuales (pesos y sesgos) de las redes principales y objetivo.
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        # Convierte los parámetros de las redes en diccionarios para una manipulación más directa.
        actor_state_dict = dict(actor_params)
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        # Actualiza los parámetros de las redes objetivo mediante un promedio ponderado (soft update, controlado por tau) entre los parámetros actuales de la red principal y los 
        # parámetros de  la red objetivo. Este enfoque asegura que las redes objetivo converjan gradualmente hacia las principales, mejorando la estabilidad del aprendizaje y 
        # reduciendo el riesgo de errores por sobreestimación de los valores Q. Un enfoque alternativo sería copiar directamente los parámetros de la red principal en la red objetivo 
        # (hard update), pero esto puede generar fluctuaciones bruscas y menor estabilidad en el entrenamiento. 
        # Cuando el algoritmo esté "bien entrenado" las redes objetivo se acercarán bastante a las principales (check).
        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau * critic_1_state_dict[name].clone() + \
                                        (1 - tau) * target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau * critic_2_state_dict[name].clone() + \
                                        (1 - tau) * target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                    (1 - tau) * target_actor_state_dict[name].clone()

        # Carga los nuevos parámetros actualizados en las redes objetivo correspondientes.
        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)


    def save_models(self):
        # Guarda los checkpoints de todas las redes (principales y objetivo) en sus respectivos directorios. Un checkpoint es un estado guardado del modelo en un momento 
        # dado del entrenamiento. Incluye principalmente los parámetros y pesos del modelo que han sido aprendidos hasta ese punto.
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        try:
            # Intenta cargar los checkpoints de las redes principales y objetivo.
            self.actor.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            self.target_critic_1.load_checkpoint()
            self.target_critic_2.load_checkpoint()
            print("Successfully loaded models")
        except:
            # Si ocurre un error, imprime un mensaje e inicia desde cero.
            print("Failed to load models. Starting from scratch")
