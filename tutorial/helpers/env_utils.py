'''
Miscellaneous functions
'''

# import warnings
# warnings.filterwarnings("ignore")

from dataclasses import asdict
import numpy as np 

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register

from environments.door_crossing import DoorCrossing
from environments.wrappers import FullyObservable, PartiallyObservable, SubGoalReward
from models.world_model import WorldModel
from agents.abstract import Agent
from confs.definitions import WorldModelEnvConfig

def load_env(config: WorldModelEnvConfig, 
				config_exclusions={'fully_observable'}):
	if config.agent_view_size != 5 and not config.compute_env_model:
		raise NotImplementedError('Agent view size != {} not implemented.'.format(config.agent_view_size))
	
	if config.compute_env_model and config.agent_view_size > 5:
		raise NotImplementedError('Computing T and O with agent_view_size > 3 would lead to a potantially very large value of possible obervations: num_possible_entities^({})'.format(config.agent_view_size*config.agent_view_size))

	if config.nondeterministic and config.robot_model_data is not None:
		validate_robot_model_data(config.robot_model_data)

	config_dict = {k: v for k, v in asdict(config).items() if k not in config_exclusions}

	env_config = {
	'name': 'GridWorld-v0',
	'config': (DoorCrossing, config_dict) }

	register(id=env_config['name'], entry_point=env_config['config'][0], kwargs=env_config['config'][1])
	env = gym.make(env_config['name'], disable_env_checker=True)
	env = TimeLimit(env, config.max_steps)

	if config.fully_observable:	
		env = FullyObservable(env)
	else: 
		env = SubGoalReward(env)
		env = PartiallyObservable(env, config.agent_view_size)

	return env

def test_env(env, agent: Agent, wm: WorldModel, save_to_path='figures/sample.gif', max_steps=10):
	'''
	Simulate a scenario 
	'''    
	
	# generate a new env
	observation, info = env.reset()
	
	# reset world model
	wm.reset()
	
	# generate initial frame
	frame = env.render()
	frames = [frame]
	
	# simulate multiple steps
	for t in range(1,max_steps):
		# robot observes and takes an action
		action = agent.act(observation)
		# step environment with action and receive new observations
		observation, _, _, _, info = env.step(action=action)
		# generate new frame
		frame = env.render()
		frames.append(frame)
	
	# save frames to gif
	# env.save_gif(frames, save_to_path=save_to_path)
	return

def validate_robot_model_data(robot_model_data):
	all_probs_valid = 	sum([i for i in robot_model_data['transitions']['forward'].values()]) and \
						sum([i for i in  robot_model_data['transitions']['left'].values()]) and \
						sum([i for i in  robot_model_data['transitions']['right'].values()])
	
	assert all_probs_valid, 'Total transition probability is not 1: {}; make sure robot_model_data values add to one'.format(all_probs_valid)
	
	corruption_rate = robot_model_data['observations']['corruption_rate']
	assert (0 <= corruption_rate <= 1), 'Sensor corruption rate is {}, it must be in [0,1] range'.format(corruption_rate)

	return  