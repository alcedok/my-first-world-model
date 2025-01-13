import torch
from torch import nn

from collections import namedtuple

from confs.definitions import TransitionModelConfig
from models.utils import gumbel_softmax

class TransitionModel(nn.Module):
	def __init__(self, config: TransitionModelConfig):
		super().__init__()
		self.config = config
		self.num_actions = config.num_actions
		self.init_gumbel_temperature = config.gumbel_temperature
		self.action_embed_dim = config.action_embed_dim
		self.fc1_hidden_dim = config.fc1_hidden_dim
		self.K = config.categorical_dim# number of categories (factors in POMDP terms)
		self.N = config.num_categorical_distributions # dimension for each factor
		self.flat_KN = self.K*self.N

		self.ActionEmbedding = nn.Embedding(num_embeddings=self.num_actions, embedding_dim=self.action_embed_dim)
		self.fc1 = nn.Linear(self.flat_KN+self.action_embed_dim, self.fc1_hidden_dim)
		self.fc2= nn.Linear(self.fc1_hidden_dim, self.fc1_hidden_dim)
		self.fc3 = nn.Linear(self.fc1_hidden_dim , self.flat_KN)
		self.act_fn = nn.ELU()

		self.ForwardOuput = namedtuple('ForwardOutput', [
													'pred_next_latent_belief', 
													'action_embed',
													'pred_next_latent_belief_logits'])

	def forward(self, belief, action, temp, hard=False):
		self.gumbel_temperature = temp
		
		# embed action
		action_embed = self.ActionEmbedding(action)
		# predict next state using transition
		state_action = torch.cat([belief.squeeze(dim=(-1,-2)), action_embed], dim=1)
		
		h = self.act_fn(self.fc1(state_action))
		h = self.act_fn(self.fc2(h))
		pred_next_latent_belief_logit = self.act_fn(self.fc3(h))

		# reshape output from [B, F] -> [B, F, 1, 1]
		batch_size, latent_state_size = pred_next_latent_belief_logit.shape[0], pred_next_latent_belief_logit.shape[1]
		pred_next_latent_belief_logit = pred_next_latent_belief_logit.view(-1, self.N, self.K)
		pred_next_latent_belief = gumbel_softmax(pred_next_latent_belief_logit.view(batch_size, latent_state_size, 1, 1), N=self.N, K=self.K, temp=self.gumbel_temperature)
		
		return self.ForwardOuput(
							pred_next_latent_belief, 
						   	action_embed, 
						  	pred_next_latent_belief_logit)