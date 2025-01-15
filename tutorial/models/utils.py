import math
import numpy as np
from collections import Counter

import torch
import torch.distributions as dist
import torch.nn.functional as F

def compute_class_weights(experience_buffer):
	objects = []
	# collect all objects from observations
	for sample in experience_buffer:
		obs_img_obj = sample.observation['observation'].flatten().tolist()
		next_obs_img_obj = sample.next_observation['observation'].flatten().tolist()
		objects.extend(obs_img_obj)
		objects.extend(next_obs_img_obj)

	# count object frequencies 
	obj_count = Counter(objects)

	# initialize class weights
	proposed_weights = [1.0]*11

	# compute inverse frequency ratio'ed by the max count
	max_class_count = obj_count[max(obj_count, key=obj_count.get)]
	for obj in obj_count.keys():
		w = max_class_count/obj_count[obj]
		proposed_weights[obj] = proposed_weights[obj]*w

	# proposed trick by taking square root 
	proposed_class_weights = torch.tensor(np.sqrt(proposed_weights).tolist())
	return proposed_class_weights

		
def world_model_loss(next_observation, observation, rewards,
					 world_model_output,
					 proposed_class_weights=None, 
					 device='cpu'):
	
	##--------------------------- 
	# observation loss components
	cross_entropy_next_obs = compute_ce_loss(world_model_output.observation_model_output.recon_next_obs_logits,
										   next_observation, weights=proposed_class_weights)
	cross_entropy_obs = compute_ce_loss(world_model_output.observation_model_output.recon_obs_logits, 
									 observation, weights=proposed_class_weights)

	latent_belief = world_model_output.observation_model_output.latent_belief.permute(0,2,1,3).squeeze(-1)
	next_latent_belief = world_model_output.observation_model_output.next_latent_belief.permute(0,2,1,3).squeeze(-1)
	# kld between the predicted latent and the learned prior
	kld_vae = compute_learned_prior_divergence(world_model_output.observation_model_output.latent_state_logits,  
											world_model_output.observation_model_output.prior_logits)

	##--------------------------- 
	## transition loss components
	pred_next_latent_belief = world_model_output.transition_model_output.pred_next_latent_belief.permute(0,2,1,3).squeeze(-1)
	pred_latent_recon_loss = compute_ce_loss(world_model_output.pred_recon_next_obs_from_latent_belief, next_observation, weights=proposed_class_weights)
	
	# pred_latent_kld = compute_learned_prior_divergence(world_model_output.transition_model_output.pred_next_latent_belief_logits, world_model_output.observation_model_output.next_latent_state_logits)
	pred_latent_kld = belief_state_kl_divergence(next_latent_belief.detach(), pred_next_latent_belief)
	# prior_latent_kld = compute_learned_prior_divergence(world_model_output.transition_model_output.pred_next_latent_belief_logits, 
	# 												world_model_output.observation_model_output.next_prior_logits)
	
	##--------------------------- 
	## reward loss components
	reward_loss = F.mse_loss( world_model_output.reward_model_output.reward.squeeze(), rewards.squeeze())

	##---------------------------
	# Total 
	ce_total = (cross_entropy_next_obs + cross_entropy_obs)
	kl_o_total = kld_vae
	kl_t_total = pred_latent_kld #+ prior_latent_kld

	return ce_total, (kl_t_total, pred_latent_recon_loss), kl_o_total, reward_loss

def compute_learned_prior_divergence(q_logits, p_logits):
	# p: predicted
	# q: target
	B, N, K = q_logits.shape

	q_logits = q_logits.view(B*N, K)
	p_logits = p_logits.view(B*N, K)
	
	q_probs = torch.softmax(q_logits, dim=-1)
	p_probs = torch.softmax(p_logits, dim=-1)

	q = dist.Categorical(probs=(q_probs))
	p = dist.Categorical(probs=(p_probs))

	# kl is of shape [B*N]
	kl = dist.kl.kl_divergence(q, p) 
	kl = kl.view(B, N)
	
	kl_loss = torch.mean(
				torch.sum(kl, dim=1))
	return kl_loss

def compute_ce_loss(recon_obs_logits, obs, weights=None, use_focal_loss=False, alpha=1, gamma=1):
	obs_target = obs.flatten(start_dim=1).long()
	obs_pred = recon_obs_logits.flatten(start_dim=2)
	if use_focal_loss:
		'''
		alpha: weighting factor for class imbalance.	
		gamma:  focusing parameter that adjusts the rate at which easy examples are down-weighted.
		'''
		cross_entropy_obs = F.cross_entropy(obs_pred, obs_target, reduction='none', weight=weights)
		pt = torch.exp(-cross_entropy_obs)
		focal_loss = alpha * (1 - pt) ** gamma * cross_entropy_obs
		return focal_loss.mean()
	else:
		cross_entropy_obs = F.cross_entropy(obs_pred, obs_target, reduction='mean', weight=weights)
	
	return cross_entropy_obs

def compute_uniform_prior_divergence(phi, device='cpu'):
	# phi is logits of shape [B, N, K] where B is batch, N is number of categorical distributions, K is number of classes
	# in our case it is of the shape [B,1,8]
	B, N, K = phi.shape
	phi = phi.view(B*N, K)
	q = dist.Categorical(probs=(phi+1e-20))
	# uniform bunch of K-class categorical distributions
	p = dist.Categorical(probs=torch.full((B*N, K), 1.0/K).to(device))
	# kl is of shape [B*N]
	kl = dist.kl.kl_divergence(q, p) 
	kl = kl.view(B, N)
	
	kl_loss = torch.mean(
				torch.sum(kl, dim=1))
	return kl_loss

def belief_state_kl_divergence(belief, pred_belief):
	# phi is logits of shape [B, N, K] where B is batch, N is number of categorical distributions, K is number of classes
	# in our case it is of the shape [B,1,8]
	B, N, K = belief.shape
	pred_belief = pred_belief.view(B*N, K)
	belief = belief.view(B*N, K)
	# predicted distribution
	q = dist.Categorical(probs=(pred_belief+1e-20))
	# target distribution
	p = dist.Categorical(probs=(belief+1e-20))
	# kl is of shape [B*N]
	# kl = dist.kl.kl_divergence(q, p) 
	kl = dist.kl.kl_divergence(p, q) 
	kl = kl.view(B, N)
	
	kl_loss = torch.mean(
				torch.sum(kl, dim=1))
	return kl_loss

def count_correct_pred(actual_obs, recon_obs_logits):
	# logits -> probs, and reshape
	obs_recon_probs = F.softmax(recon_obs_logits.flatten(start_dim=2), dim=1)
	# probs -> label (highest prob), and reshape
	obs_recon_pred_labels = torch.max(obs_recon_probs, dim=1, keepdim=True).indices.squeeze(1)
	# element-wise equality check
	obs_eq = torch.eq(actual_obs.flatten(start_dim=1).long().squeeze(), obs_recon_pred_labels)
	# sum correct 
	correct_obs_recon = torch.sum(obs_eq)
	return correct_obs_recon, obs_recon_pred_labels, obs_eq

def training_reconstruction_accuracy(next_observation, observation, world_model_output):
	# next obs recon
	correct_next_obs_recon_batch, _, _ = count_correct_pred(next_observation, world_model_output.observation_model_output.recon_next_obs_logits)
	# obs recon
	correct_obs_recon_batch, _, _ =  count_correct_pred(observation, world_model_output.observation_model_output.recon_obs_logits)
	# pred_next_obs recon (using transition from obs and action)
	correct_pred_obs_recon_batch, _, _ =  count_correct_pred(next_observation, world_model_output.pred_recon_next_obs_from_latent_belief)

	return correct_next_obs_recon_batch, correct_obs_recon_batch, correct_pred_obs_recon_batch

def gumbel_softmax(latent_belief, N, K, temp, gumbel_hard=False):
		# latent_state is of shape: [B batch, N*K feature, 1, 1]
		# reshape to be able to sample N distributions 
		latent_belief_NK = latent_belief.view(-1, N, K)
		# take the gumbel-softmax on the last dimension K (dim=-1)
		latent_belief_categorical_NK = F.gumbel_softmax(latent_belief_NK, tau=temp, hard=gumbel_hard, dim=-1)
		# reshape back to original shape
		latent_belief_categorical = latent_belief_categorical_NK.view(latent_belief.shape)
		return latent_belief_categorical

def compression_factor(image_shape: tuple, latent_shape: tuple):
	'''
	compression factor of representation
	'''
	numerator = math.prod(image_shape)
	denominator = math.prod(latent_shape)
	return numerator/denominator

def data_to_tensors(observation, action, next_observation, rewards, device='cpu'):

	''' convert data to tensors if not already'''

	def add_batch_dim(tensor, required_dim=4):
		#TODO: not fully implemented, needs to know the required dim for each
		''' check the tensor has the required number of dimensions:'''
		# if missing the batch dimensions
		if tensor.dim() == required_dim - 1:
			tensor = tensor.unsqueeze(0)  # Add batch dimension at the front
		return tensor
	
	def to_tensor(data, dtype, device):
		if not torch.is_tensor(data):
			return torch.tensor(data, dtype=dtype, device=device)
		return data.to(dtype=dtype, device=device)

	observation_tensor = to_tensor(observation['observation'], torch.int, device)
	action_tensor = to_tensor(action, torch.int, device)
	next_observation_tensor = to_tensor(next_observation['observation'], torch.int, device)
	rewards_tensor = to_tensor(rewards, torch.float, device)

	# observation_tensor = observation['observation'].to(torch.int).to(device)
	# action_tensor = action.to(torch.int).to(device)
	# next_observation_tensor = next_observation['observation'].to(torch.int).to(device)
	# rewards_tensor = rewards.to(torch.float).to(device)

	return (observation_tensor, action_tensor, next_observation_tensor, rewards_tensor)

def uniform_latent_sampler(latent_shape: tuple, temperature: float, hard:bool = False):
	''' sample from a uniform prior categorical distribution using gumbel-softmax '''
	if not isinstance(latent_shape, tuple) or len(latent_shape) != 3:
		raise ValueError('Expected shape to be a tuple of length 3 (B, N, K), received {} '.format(latent_shape))
	(B, N, K) = latent_shape
	uniform_prior = torch.zeros(B,N,K)
	sample = F.gumbel_softmax(uniform_prior, tau=temperature, hard=hard, dim=-1)
	return sample

def learned_prior_sampler(prior_logits: torch.Tensor, temperature: float, hard: bool = False):
    ''' sample from a learned prior categorical distribution using gumbel-softmax
		prior logits of the learned prior distribution, shape (B, N, K); used for context
	'''
    if not isinstance(prior_logits, torch.Tensor) or len(prior_logits.shape) != 3:
        raise ValueError('Expected prior_logits to be a tensor of shape (B, N, K), received {}'.format(prior_logits.shape))
    
    # apply Gumbel-Softmax sampling
    sample = F.gumbel_softmax(prior_logits, tau=temperature, hard=hard, dim=-1)
    return sample