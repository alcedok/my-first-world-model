import numpy as np
import random 
from operator import itemgetter

from minigrid.core.constants import OBJECT_TO_IDX

from agents.abstract import Agent
from helpers.metrics_utils import TrainingCallback
from environments.constants import Directions

class QLearningAgent(Agent):
	'''
	Agent that implements the Q-Learning policy
	It is currently imeplemnted for MDP environments with a defined Transition and Observation model
	'''
	def __init__(self, 
					policy,
					model_metrics_callback: TrainingCallback,
					exploit_after_learning=False,			 
					agent_name='Q-Learning Agent'):
		
		self.policy = policy
		self.model_metrics_callback = model_metrics_callback
		self.valid_actions = self.policy.valid_actions
		self.agent_name = agent_name
		self.exploit_after_learning = exploit_after_learning

	def reset(self, observation=None):
		return 
	
	def act(self, observation):
		state_id = self.process_obs(observation)
		# explore: random action
		if (random.random() < self.policy.epsilon) and (not self.exploit_after_learning):
			action = random.choice(list(self.valid_actions))
			value = self.policy.Q_lookup[(state_id, action)]
		# exploit: choose action with max q-value
		else:
			action_value_options = [ (action_i, self.policy.Q_lookup[(state_id, action_i)]) for action_i in self.valid_actions]
			action, value = max(action_value_options, key=itemgetter(1))
		return action 
	
	def process_obs(self, observation):
		grid_state = observation['state'] # we transpose to match MiniGrid representation
		orientation = observation['agent_direction']
		state_tuple = self.get_state_tuple(grid_state, orientation)
		state_id = self.policy.state_rlookup[state_tuple]
		return state_id
	
	def get_state_tuple(self, grid_state, orientation):
		''' return (x,y,orientation) tuple required to lookup state_id '''
		if not isinstance(orientation, Directions):
			raise ValueError('Expected orientation to be of type Enum Directions')
		agent_pos_x, agent_pos_y = np.argwhere(grid_state == OBJECT_TO_IDX['agent'])[0]
		orientation = orientation
		return (agent_pos_y, agent_pos_x, orientation)
	
	def update(self, observation, action, reward, next_observation, terminated):	
		state_id = self.process_obs(observation)
		next_state_id = self.process_obs(next_observation)
		self.policy.update_q_value(state_id, action, reward, next_state_id, terminated)