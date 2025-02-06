import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RewardWrapper
from minigrid.core import constants 
from minigrid.core.actions import Actions
from environments.constants import Directions
import numpy as np 

class PartiallyObservable(gym.ObservationWrapper):
	'''
	Egocentric, partially observable gridworld.

	Sensing range defined by 'agent_view_size', 
	where the agent only perceives in-front and to the sides, 
	see diagram below for agent_view_size=3:
	
	+---+---+---+
	|   |   |   |
	+---+---+---+
	|   |   |   |
	+---+---+---+
	|   | ▲ |   |
	+---+---+---+
	 
	'''

	def __init__(self, env, agent_view_size=3):
		super().__init__(env)
		assert agent_view_size % 2 == 1
		assert agent_view_size >= 3

		self.agent_view_size = agent_view_size
	
		view_space = spaces.Box(
			low=0,
			high=255,
			shape=(self.env.unwrapped.width, self.env.unwrapped.height, 1),
			dtype='uint8')

		self.observation_space = spaces.Dict({
			'observation': view_space})

	def observation(self, obs):
		env = self.unwrapped
		grid, vis_mask = env.gen_obs_grid(self.agent_view_size)
		
		# encode the partially observable view into a numpy array
		# only keep object_id dimension
		view = grid.encode(vis_mask)[:,:,0]

		return {'observation': view}

class FullyObservable(gym.ObservationWrapper):
	'''
	Allocentric, fully observable gridworld using a compact grid encoding.
	
	Grid is represented as a matrix where each (row,col) corresponds
	to a position (x,y) in the grid. The value at (i,j) is the object index. 

	The global 'direction' of the agent is provided as a separate entity in the observation dict. 	
	'''

	def __init__(self, env):
		super().__init__(env)
		
		state_space = spaces.Box(
			low=0,
			high=255,
			shape=(self.env.unwrapped.width, self.env.unwrapped.height, 1),
			dtype='uint8')

		self.observation_space = spaces.Dict({
			'state': state_space,
			'agent_direction': spaces.Discrete(4)
			})

	def observation(self, obs):
		env = self.unwrapped
		# only keep object_id dimension
		state = env.grid.encode()[:,:,0]
		
		# include agent in view 
		state[env.agent_pos[1]][env.agent_pos[0]] = np.array(
			[constants.OBJECT_TO_IDX['agent']])

		return {
			'state': state, 
			'agent_direction':Directions(obs['direction'])}



class PriviledgedModelBuilder(gym.ObservationWrapper):
	'''
	Used to construct the Transition and Observation models.
	This is a priviledged wrapper since it provides both full and partial observability
	
	Allocentric, fully observable gridworld using a compact grid encoding.
	
	Grid is represented as a matrix where each (row,col) corresponds
	to a position (x,y) in the grid. The value at (i,j) is the object index. 

	The global 'direction' of the agent is provided as a separate entity in the observation dict. 	
	'''

	def __init__(self, env):
		super().__init__(env)
		
		self.fully_observable_wrapper = FullyObservable(env)
		self.partially_observable_wrapper = PartiallyObservable(env)
		
		self.observation_space = spaces.Dict({
			'full_observation': self.fully_observable_wrapper.observation_space['state'],
			'partial_observation': self.partially_observable_wrapper.observation_space['observation'],
			'agent_direction': self.fully_observable_wrapper.observation_space['agent_direction']
			})

	def observation(self, obs):
		# env = self.unwrapped
		full_obs = self.fully_observable_wrapper.observation(obs)
		partial_obs = self.partially_observable_wrapper.observation(obs)

		return {
			'full_observation': full_obs['observation'],
			'partial_observation': partial_obs['observation'],
			'agent_direction':full_obs['agent_direction']}


class SubGoalReward(RewardWrapper):
	'''
	Add reward by reaching subgoals on grid, 
	we make sure to set them as visited so the agent doens't spam the subgoal
	'''
	def __init__(self, env):
		super().__init__(env)
		self.visited = False 
		self.goal_region_set = None
		self.init_region_set = None
		self.agent_init_region = None

	def reset(self, **kwargs):
		self.visited = False
		observation, info = self.env.reset(**kwargs)
		self.goal_region_set = {c for c in self.unwrapped.goal_region} # (x,y) tuples
		self.init_region_set = {c for c in self.unwrapped.init_region}
		self._agent_init_pos = self.unwrapped.agent_pos
		return observation, info

	def reward(self, reward):
		agent_at_door = (self.unwrapped.agent_pos == self.unwrapped.door_position)
		agent_started_in_goal_region = self._agent_init_pos in self.goal_region_set
		if (agent_at_door) and not (agent_started_in_goal_region) and not self.visited:
			reward += 0.5
			self.visited = True
		elif (agent_at_door) and (agent_started_in_goal_region):
			reward = 0
		
		return reward
