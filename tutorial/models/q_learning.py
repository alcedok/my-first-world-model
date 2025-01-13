from collections import defaultdict
import numpy as np
import random, time
from environments.env_models import EnvModel

class QLearning:
	def __init__(self, env_model: EnvModel, 
			  learning_rate=0.2, 
			  discount=0.99, 
			  randomize_q=False, 
			  epsilon=0.1):
		'''
		epsilon-greedy Q-Learning
		- discount: The discount factor gamma range [0,1]
		- assumes T imeplmented in EnvModel.TransitionModel to have shape T[state_id, action, next_state_id]
		'''

		self.env_models = env_model
		self.states = env_model.transition_model.states
		self.valid_actions =  env_model.valid_actions
		self.state_rlookup = env_model.transition_model.rlookup
		self.state_ids = [s_id for s_id in env_model.transition_model.lookup.keys()]    

		self.num_states = len(self.state_ids) 
		self.num_actions = env_model.num_actions

		self.learning_rate = learning_rate
		self.discount = discount       # discount factor       
		self.epsilon = epsilon          	

		self.randomize_q = randomize_q

		self.Q_lookup = self.init_Q(randomize_q)
	
	def init_Q(self, randomize=False):
		''' initialize Q table with values, zero default, random optional '''
		if randomize:
			for state_i in self.state_ids:
				for action_i in self.valid_actions:
					Q_lookup[(state_i, action_i)] = random.uniform(0.0,20.0)
		else:
			Q_lookup = defaultdict(lambda: 0.0)    
		return Q_lookup
	
	def update_q_value(self, state_id, action, reward, next_state_id, terminated):
		max_next_q_value = max(self.Q_lookup[(next_state_id, action_i)] for action_i in self.valid_actions)
		self.Q_lookup[(state_id, action)] += self.learning_rate * (reward + (self.discount * (1 - terminated)) * max_next_q_value - self.Q_lookup[(state_id, action)])