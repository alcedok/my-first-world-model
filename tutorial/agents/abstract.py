from abc import ABC, abstractmethod

class Agent(ABC):
	'''
	Abstract class used to implement tutorial's agents
	'''
	def __init__(self, 
				 valid_actions,
				 agent_name, *kwargs):
		self.valid_actions = valid_actions
		self.agent_name = agent_name
		
	@abstractmethod
	def reset(self, observations=None):
		return 
	
	@abstractmethod
	def act(self, observations=None):
		return 
	