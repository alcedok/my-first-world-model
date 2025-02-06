from collections import namedtuple

import torch
from torch import nn

from confs.definitions import ObservationModelConfig
from models.utils import gumbel_softmax

class ObservationModel(nn.Module):
	def __init__(self, config: ObservationModelConfig, device='cpu'):
		super().__init__()
		self.config = config
		self.init_gumbel_temperature = config.gumbel_temperature
		self.concept_dim = config.concept_dim
		self.concept_embed_dim = config.concept_embed_dim
		self.num_att_heads = config.num_att_heads
		self.K = config.categorical_dim # number of categories (factors in POMDP terms)
		self.N = config.num_categorical_distributions # dimension for each factor
		self.categorical_dim = self.K*self.N
		self.flat_KN = self.K*self.N
		self.conv1_hidden_dim =  config.conv1_hidden_dim
		self.prior_fc1_hidden_dim = 100
		self.device = device 

		# concept (object) in grid to vector embedding
		self.obs_to_concept_embedding = nn.Embedding(num_embeddings=self.concept_dim, embedding_dim=self.concept_embed_dim)
		
		# encoder
		self.enc_conv_1= nn.Conv2d(self.concept_embed_dim, self.conv1_hidden_dim, kernel_size=3, stride=1, padding=0)
		self.enc_conv_2= nn.Conv2d(self.conv1_hidden_dim, self.categorical_dim, kernel_size=3, stride=1, padding=0)
		self.enc_att_1 = nn.MultiheadAttention(embed_dim=self.conv1_hidden_dim, num_heads=self.num_att_heads)

		# decoder
		self.dec_conv1_1 = nn.ConvTranspose2d(self.categorical_dim, self.conv1_hidden_dim, kernel_size=3, stride=1, output_padding=0, padding=0)
		self.dec_conv1_2 = nn.ConvTranspose2d(self.conv1_hidden_dim, self.concept_dim, kernel_size=3, stride=1, output_padding=0, padding=0)
		
		# learned prior
		self.prior_fc1 =  nn.Linear(self.N, self.prior_fc1_hidden_dim)
		self.prior_fc2 = nn.Linear(self.prior_fc1_hidden_dim, self.flat_KN)

		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()
		self.act_fn = nn.ELU()

		self.temp = self.init_gumbel_temperature

		self.ForwardOuput = namedtuple('ForwardOutput', [
													'recon_next_obs_logits', 'recon_obs_logits', 
													'next_latent_belief', 'latent_belief',
													'prior_logits', 'next_prior_logits',
													'latent_state_logits', 'next_latent_state_logits'])


	def encode(self, observation):
		obs_emb = self.obs_to_concept_embedding(observation).permute(0,3,1,2)
		h = self.act_fn(self.enc_conv_1(obs_emb))

		batch_size, channels, height, width = h.size()
		h = h.view(batch_size, channels, height * width).permute(2, 0, 1)  # (sequence_length, batch_size, embedding_dim) for attention
		attn_output, _ = self.enc_att_1(h, h, h)
		attn_output = attn_output.permute(1, 2, 0).contiguous().view(batch_size, channels, height, width)

		latent_belief = self.act_fn(self.enc_conv_2(attn_output))

		return obs_emb, latent_belief
	
	def decode(self, latent_belief):
		h1 = self.act_fn(self.dec_conv1_1(latent_belief))
		recon_obs_logits = self.dec_conv1_2(h1)
		return recon_obs_logits
	
	def prior(self, context):
		''' input to the prior network could be used to bias the categories, or to provide context on what to sample'''
		h = self.act_fn(self.prior_fc1(context))
		prior_categorical_logits = self.act_fn(self.prior_fc2(h))
		return prior_categorical_logits
	
	def get_belief(self, observation, gumbel_hard=False):
		obs_emb, latent_belief = self.encode(observation.to(torch.int).detach())
		# reshape to be able to sample N distributions 
		latent_belief_categorical = gumbel_softmax(latent_belief, N=self.N, K=self.K, temp=self.temp, gumbel_hard=gumbel_hard)
		belief_as_ints = torch.argmax(latent_belief_categorical.squeeze().view(-1,self.N,self.K), dim=-1).tolist()[0]
		return obs_emb, latent_belief, latent_belief_categorical, tuple(belief_as_ints)

	def get_recon(self, latent_belief):
		return torch.max(latent_belief, dim=1, keepdim=True).indices.squeeze()
	
	def forward(self, observation, next_observation, temp, gumbel_hard=False):
		self.temp = temp 

		# obs and next_obs to latent-state
		obs_emb, latent_state_logits = self.encode(observation)
		next_obs_emb, next_latent_state_logits = self.encode(next_observation)
		
		next_latent_belief = gumbel_softmax(next_latent_state_logits, N=self.N, K=self.K, temp=self.temp, gumbel_hard=gumbel_hard)
		latent_belief = gumbel_softmax(latent_state_logits, N=self.N, K=self.K, temp=self.temp, gumbel_hard=gumbel_hard)

		recon_next_obs_logits = self.decode(next_latent_belief)
		recon_obs_logits = self.decode(latent_belief)

		# logits using prior 
		prior_logits = self.prior(torch.zeros(latent_belief.size(0), self.N, device=self.device))
		prior_logits = prior_logits.view(-1, self.N, self.K)

		next_prior_logits = self.prior(torch.zeros(next_latent_belief.size(0), self.N, device=self.device))
		next_prior_logits = next_prior_logits.view(-1, self.N, self.K)

		# reshape to match prior output shape
		latent_state_logits = latent_state_logits.view(-1, self.N, self.K)
		next_latent_state_logits = next_latent_state_logits.view(-1, self.N, self.K)

		return self.ForwardOuput(recon_next_obs_logits, recon_obs_logits, 
						   next_latent_belief, latent_belief, 
						   prior_logits, next_prior_logits, 
						   latent_state_logits, next_latent_state_logits)