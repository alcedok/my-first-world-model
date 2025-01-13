from collections import defaultdict
import numpy as np
import random, time
from environments.env_models import EnvModel

class ValueIteration:
	def __init__(self, env_model: EnvModel, discount=0.99, tol=1e-6):
		'''
		Value Iteration algorithm.
		- discount: The discount factor gamma range [0,1]
		- theta: The threshold for convergence.
		- assumes T imeplmented in EnvModel.TransitionModel to have shape T[state, action, next_state]
		'''

		self.env_models = env_model
		self.T = env_model.transition_model.T
		self.states = env_model.transition_model.states
		self.reward_model = env_model.reward_model
		self.state_rlookup = env_model.transition_model.rlookup
		self.state_ids = [s_id for s_id in env_model.transition_model.lookup.keys()]    
		self.num_states = len(self.state_ids) 
		self.num_actions = env_model.num_actions
		self.discount = discount       # discount factor       
		self.tol = tol          	   # convergence tolerance threshold            

	def run(self):
		'''
		run value iteration to compute the optimal policy.
		
		outputs:
			V: A 1D numpy array where V[state_id] is the value of the corresponding state.
			policy: A 1D numpy array where policy[state_id] gives the optimal action int for that state.
		'''

		# initialize value function (V), since r_min is zero for our scenarios
		V = np.zeros(self.num_states)
		# initialize optimal policy
		policy = np.zeros(self.num_states, dtype=int)  
		
		iters = 0
		while True:
			delta = 0
			for s in self.state_ids:
				# compute the expected value of taking each action from current state
				action_values = np.zeros(self.num_actions)
				for a in range(self.num_actions):
					action_values[a] = np.sum(self.T[s, a, :] * V)				
				# Bellman equation
				new_value = self.reward_model[s] + self.discount * np.max(action_values)
				# update the largest change (delta) for convergence
				delta = max(delta, abs(new_value - V[s]))
				V[s] = new_value	
				iters += 1	
			# check for convergence
			if delta < self.tol:
				break
		
		print('Value Iteration converged at iters: {:,}'.format(iters))

		# get optimal policy
		for s in self.state_ids:
			action_values = np.zeros(self.num_actions)
			# get value for each action
			for a in range(self.num_actions):
				action_values[a] = np.sum(self.T[s, a, :] * V)
			# optimal action is the one with the highest value
			policy[s] = np.argmax(action_values)  

		return V, policy