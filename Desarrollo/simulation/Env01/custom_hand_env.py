import robosuite as suite
from robosuite.models.robots import MujocoRobot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string
import numpy as np
import gym
from gym import spaces

class RoboticHand(MujocoRobot):
    """Custom robotic hand model"""
    def __init__(self):
        super().__init__("robotic_hand")
        
        # Define robot parameters
        self.n_joints = 15  # Total number of joints (3 per finger x 5 fingers)
        self.joint_names = [
            # Índice
            "joint_indice_0", "joint_indice_1", "joint_indice_2",
            # Medio
            "joint_medio_0", "joint_medio_1", "joint_medio_2",
            # Anular
            "joint_anular_0", "joint_anular_1", "joint_anular_2",
            # Meñique
            "joint_menique_0", "joint_menique_1", "joint_menique_2",
            # Pulgar
            "joint_pulgar_0", "joint_pulgar_1"
        ]
        
        # Joint limits from SDF
        self.joint_limits = {
            "joint_indice_1": {"lower": -0.95993, "upper": 0},
            "joint_indice_2": {"lower": 0, "upper": 1.1868},
            # Other joints default to [-π, π] if not specified
        }

class RoboticHandEnv(gym.Env):
    def __init__(self):
        self.hand = RoboticHand()
        
        # Initialize robosuite environment
        self.env = suite.make(
            "Empty",  # Empty environment as base
            robots=[self.hand],
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            control_freq=20,
            horizon=500,
            use_camera_obs=False,
        )
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.hand.n_joints,),
            dtype=np.float32
        )
        
        # Observation space includes joint positions and velocities
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.hand.n_joints * 2,),  # positions + velocities
            dtype=np.float32
        )

    def reset(self):
        """Reset environment to initial state"""
        obs = self.env.reset()
        return self._get_obs()

    def step(self, action):
        """Execute action and return new state"""
        # Scale actions from [-1, 1] to actual joint limits
        scaled_action = self._scale_action(action)
        
        # Execute action
        obs, reward, done, info = self.env.step(scaled_action)
        
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """Get current observation"""
        robot_states = self.env.robots[0].get_robot_state()
        
        # Combine joint positions and velocities
        obs = np.concatenate([
            robot_states['joint_pos'],
            robot_states['joint_vel'],
        ])
        
        return obs

    def _scale_action(self, action):
        """Scale actions from [-1, 1] to actual joint limits"""
        scaled_action = np.zeros_like(action)
        for i, joint_name in enumerate(self.hand.joint_names):
            if joint_name in self.hand.joint_limits:
                limits = self.hand.joint_limits[joint_name]
                scaled_action[i] = (
                    (action[i] + 1) / 2 * 
                    (limits["upper"] - limits["lower"]) + 
                    limits["lower"]
                )
            else:
                scaled_action[i] = action[i] * np.pi  # Default to [-π, π]
                
        return scaled_action

    def render(self, mode='human'):
        """Render environment"""
        return self.env.render()

    def close(self):
        """Clean up resources"""
        self.env.close() 