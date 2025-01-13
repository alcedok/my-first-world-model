import numpy as np
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from confs.definitions import DynaQConfig

class QNetwork(nn.Module):
	def __init__(self, input_dim, hidden_dim):
		super().__init__()

		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, 1)
		self.ForwardOutout = namedtuple('ForwardOutput', ['q_value'])

	def forward(self, state, action):
		x = torch.cat([state, action], dim=1)
		x = torch.relu(self.fc1(x))
		q_value = self.fc2(x)
		return self.ForwardOutout(q_value)

class DynaQ(nn.Module):
	def __init__(self, config: DynaQConfig, action_embed_fn: nn.Embedding, device='cpu'):
		super().__init__()
		self.device = device
		self.ActionEmbedding = action_embed_fn
		self.qnet_input_dim = config.qnet_input_dim
		self.qnet_fc1_hidden_dim = config.qnet_fc1_hidden_dim

		self.valid_actions = config.valid_actions
		self.valid_actions_tensor = self._valid_actions_to_tensors()
		self.num_actions = config.num_actions
		self.gamma = config.gamma
		self.epsilon = config.epsilon
		self.alpha = config.optimizer_learning_rate

		self.num_simulations = config.num_simulations
		
		# Q-network and optimizer
		self.q_network = QNetwork(self.qnet_input_dim, self.qnet_fc1_hidden_dim)
		self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)

		self.LossOutput = namedtuple('LossOutput', ['total_loss'])
		
	def _valid_actions_to_tensors(self):
		''' convert list of valid actions into a batch tensor (batch, num_actions) where batch_dim = 1 '''
		return torch.stack(
			[torch.tensor(np.array(action)).to(torch.int).to(self.device) 
			for action in self.valid_actions])
	
	def forward(self, state, action, embed_actions):
		#TODO: need to improve this to be less hacky
		if embed_actions:
			action = self.ActionEmbedding(action)
		
		state_dim, action_dim  = state.shape[0], action.shape[0]
		if state_dim != action_dim:
			# raise ValueError('unmatched batch dimension for state with dim {} and action dim {}, they should be equal'.format(state_dim, action_dim))
			# print('unmatched batch dimension for state with dim {} and action dim {}, they should be equal'.format(state_dim, action_dim))
			if state_dim > action_dim:
				# We are testing actions across many states, so repeat actions to match state_dim
				repeat_dim = int(state_dim/self.num_actions)
				action = action.repeat(repeat_dim, 1) #TODO: this is hacky, need to find a better way of knowing how to repeat
				# print('We are testing actions across many states, so repeat actions to match state_dim')
			else:
				# We are testing many actions on one state, so repeat_interleaved states to match action_dim
				state = state.repeat_interleave(action_dim, dim=0)
				# print('We are testing many actions on one state, so repeat_interleaved states to match action_dim')
		# otherwise we are testing one action to one state, do nothing
		q_net_forward_output = self.q_network(state, action)
		return q_net_forward_output
	
	def choose_action(self, state):
		if random.random() < self.epsilon:
			return random.randint(0, self.num_actions - 1)
		with torch.no_grad():
			forward_output = self.forward(state, self.valid_actions_tensor, embed_actions=True)
			return torch.argmax(forward_output.q_value, dim=0).item()

def update(model, optimizer, inputs, embed_actions):
	''' update Q-function '''
	optimizer.zero_grad()
	(state, next_state, action, reward, terminated) = inputs

	# compute target
	with torch.no_grad():

		forward_output = model(next_state, model.valid_actions_tensor, embed_actions=True)

		next_q_values  = forward_output.q_value
		# maximum Q-value across all actions
		max_next_q_value = torch.max(next_q_values).item()
		# compute target Q
		target = reward + model.gamma * (1 - terminated) * max_next_q_value
	
	# Compute loss and update
	forward_output = model(state, action, embed_actions=embed_actions)
	
	loss = F.mse_loss(forward_output.q_value.squeeze(), target.squeeze(), reduction='mean')
	loss.backward()
	optimizer.step()
	loss_output = model.LossOutput(loss)
	return loss_output, forward_output