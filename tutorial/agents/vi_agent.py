import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX

from agents.abstract import Agent
from helpers.metrics_utils import TrainingCallback
from environments.constants import Directions
from environments.env_models import EnvModel

class ValueIterationAgent(Agent):
	'''
	Agent that implements the Value Iteration policy
	It is currently imeplemnted for MDP environments with a defined Transition and Observation model
	'''
	def __init__(self, 
					env_model: EnvModel,
					policy,
					model_metrics_callback: TrainingCallback,			 
					agent_name='VI Agent'):
		
		self.env_model = env_model
		self.policy = policy
		self.valid_actions = self.env_model.valid_actions
		self.state_id_lookup = self.env_model.transition_model.rlookup
		self.model_metrics_callback = model_metrics_callback
		self.agent_name = agent_name
		
	def reset(self, observation=None):
		return 
	
	def act(self, observation):
		state_id = self.process_obs(observation)
		action = self.policy[state_id]
		return action
	
	def process_obs(self, observation):
		grid_state = observation['state'] # we transpose to match MiniGrid representation
		orientation = observation['agent_direction']
		state_tuple = self.get_state_tuple(grid_state, orientation)
		state_id = self.state_id_lookup[state_tuple]
		return state_id
	
	def get_state_tuple(self, grid_state, orientation):
		''' return (x,y,orientation) tuple required to lookup state_id '''
		if not isinstance(orientation, Directions):
			raise ValueError('Expected orientation to be of type Enum Directions')
		agent_pos_x, agent_pos_y = np.argwhere(grid_state == OBJECT_TO_IDX['agent'])[0]
		orientation = orientation
		return (agent_pos_y, agent_pos_x, orientation)