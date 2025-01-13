import random 

from agents.abstract import Agent

class RandomAgent(Agent):
	def __init__(self, valid_actions, agent_name='random-agent'):
		self.valid_actions = valid_actions
		self.policy = lambda: random.choice(list(self.valid_actions))
		super().__init__(self.valid_actions, agent_name)
	
	def reset(self, observations=None):
		return 
	
	def act(self, observations=None):
		return self.policy()