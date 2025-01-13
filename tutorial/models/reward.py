from collections import namedtuple

import torch
from torch import nn

from confs.definitions import RewardModelConfig

class RewardModel(nn.Module):
	def __init__(self, config: RewardModelConfig):
		super().__init__()
		self.config = config
		self.latent_dim = config.categorical_dim * config.num_categorical_distributions
		self.action_embed_dim = config.action_embed_dim
		self.fc1_hidden_dim = config.fc1_hidden_dim
		self.reward_output_dim = config.reward_output_dim

		if config.with_action:
			# R(s,a)
			self.fc_1 = nn.Linear((self.latent_dim)+self.action_embed_dim, self.fc1_hidden_dim)
		else:
			# R(s)
			self.fc_1 = nn.Linear((self.latent_dim), self.fc1_hidden_dim)
			
		self.fc_2 = nn.Linear(self.fc1_hidden_dim, self.fc1_hidden_dim)
		self.fc_3 = nn.Linear(self.fc1_hidden_dim, self.reward_output_dim)

		self.act_fn = nn.ELU()

		self.ForwardOutput = namedtuple('ForwardOutput', ['reward'])

	def forward(self, state, action_emb=None):
		# we need to remove unecessary dimensions from input latent (B,K*N,1,1)
		# where B: batch, K: number of distirbutions, N: num of categories per distribution
		if action_emb is not None:
			model_inputs = torch.cat([state.squeeze(dim=(-1,-2)), action_emb], dim=1)
		else: 
			model_inputs = state.squeeze(dim=(-1,-2))
		h = self.act_fn(self.fc_1(model_inputs))
		h = self.act_fn(self.fc_2(h))
		reward = self.fc_3(h)
		return self.ForwardOutput(reward)
