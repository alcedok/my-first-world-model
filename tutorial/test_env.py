
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from dataclasses import asdict

from minigrid.core.actions import Actions

import pygame

from configs import WorldModelEnvConfig, robot_model_data
from environments.door_crossing import DoorCrossing
from environments.wrappers import PartiallyObservable, SubGoalReward

'''
Modified from minigrid.manual_control

Used to directly interact and control the agent using the pygame interface. 
It probably won't run on a notebook, so call it from terminal:

python test_env.py
'''

class ManualControl:

    def __init__(
        self,
        env: Env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.unwrapped.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)

if __name__ == "__main__":
        
    ''''
    Make changes to the configuration
    '''


    simple_config = WorldModelEnvConfig(width=7, 
                                        height=9, 
                                        agent_view_size=5, 
                                        randomize=True,
                                        randomize_agent=False,
                                        curriculum_learning_prob=0.0, 
                                        compute_env_model=False, 
                                        nondeterministic=False, 
                                        robot_model_data=None)

    config_exclusions={'fully_observable'}
    config_dict = {k: v for k, v in asdict(simple_config).items() if k not in config_exclusions}

    env_config = {
    'name': 'GridWorld-v0',
    'config': (DoorCrossing, config_dict) }

    register(id=env_config['name'], entry_point=env_config['config'][0], kwargs=env_config['config'][1])
    env = gym.make(env_config['name'], disable_env_checker=True)

    env = TimeLimit(env, max_episode_steps=simple_config.max_steps)
    env = PartiallyObservable(env, simple_config.agent_view_size)
    env = SubGoalReward(env)

    manual_control = ManualControl(env)
    manual_control.start()