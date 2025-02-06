import tqdm
import datetime, time, os 
import numpy as np
from collections import namedtuple
from dataclasses import asdict
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv

from agents.abstract import Agent
from models.observation import ObservationModel
from models.transition import TransitionModel
from models.reward import RewardModel
from confs.definitions import WorldModelTrainingConfig, WorldModelConfig
from helpers.metrics_utils import TrainingCallback, MetricTracker
from helpers.data_utils import collect_experiences
from models.utils import (world_model_loss, 
						  compute_class_weights, 
						  data_to_tensors, 
						  training_reconstruction_accuracy, 
						  uniform_latent_sampler,
						  learned_prior_sampler)

class WorldModel(nn.Module):
	def __init__(self, 
			  config: WorldModelConfig,
			  model_id=0, 
			  device='cpu'):
		super().__init__()

		#configs
		self.config = config
		self.observation_model_config = config.observation_model_config
		self.transition_model_config = config.transition_model_config
		self.reward_model_config = config.reward_model_config

		self.model_id  = model_id
		
		# hyper-parameters 
		self.temp = config.gumbel_temperature
		self.gumbel_hard = config.gumbel_hard
		self.categorical_dim = config.categorical_dim
		self.num_categorical_distributions = config.num_categorical_distributions
		self.belief_dim = self.categorical_dim*self.num_categorical_distributions
		self.proposed_class_weights = None 

		# models
		self.observation_model = ObservationModel(self.observation_model_config, device=device).to(device)
		self.transition_model = TransitionModel(self.transition_model_config).to(device)
		self.reward_model = RewardModel(self.reward_model_config).to(device)
		
		self.prev_belief = self.init_belief()
		self.belief = self.prev_belief

		self.ForwardOutout = namedtuple('ForwardOutput', ['observation_model_output', 'transition_model_output', 'reward_model_output', 'pred_recon_next_obs_from_latent_belief'])
		self.LossOutput = namedtuple('LossOutput', ['total_loss', 'observation_loss', 'transition_loss', 'reward_loss', 'observation_recon_loss', 'observation_kld'])

	def init_belief(self):
		return torch.zeros(size=(self.belief_dim, ))
	
	def set_belief(self, observation):
		observation_model_output = self.observation_model(observation)
		self.belief = observation_model_output.latent_belief

	def step(self, state, action):
		if action not in self.valid_actions:
			raise ValueError('Expected input action of type {}; Received: {}'.format(Actions, action))
		# estimate new belief
		new_belief = self.transition_model(self.belief, action, self.temp)
		# estimate reward given the new belief
		reward = self.reward_model(new_belief)
		# update beliefs
		self.prev_belief = self.belief
		self.belief = new_belief
		return self.belief, reward

	def reset(self):
		self.prev_belief = self.init_belief()

	def render(self, observation):
		'''
		Visualize: observation, latent, reconstruction
		'''
		observation = torch.tensor(observation['observation']).to(torch.int).unsqueeze(0)
		bs_emb, latent_belief, latent_belief_categorical, belief_as_ints = self.observation_model.get_belief(observation)

		return    

	def freeze_weights(self):
		all_internal_models  = nn.ModuleList([self.observation_model, self.transition_model, self.reward_model])
		for model in all_internal_models:
			# set to eval
			model.eval()
			for param in model.parameters():
				# freeze weights
				param.requires_grad = False 

	def forward(self, observation, next_observation, action, temp, gumbel_hard=False):
		# observation components
		observation_model_output = self.observation_model(observation, next_observation, temp=temp, gumbel_hard=gumbel_hard)
		belief = observation_model_output.latent_belief

		# transition components
		transition_model_output = self.transition_model(belief, action, temp)
		next_belief = transition_model_output.pred_next_latent_belief
		
		# reward components
		reward_model_output = self.reward_model(belief)

		# reconstruct the predicted next_belief 
		pred_recon_next_obs_from_latent_belief = self.observation_model.decode(next_belief)

		world_model_output = self.ForwardOutout(observation_model_output, transition_model_output, reward_model_output, pred_recon_next_obs_from_latent_belief)
		return world_model_output
	
	@classmethod
	def warm_up(cls, 
			 model_config: WorldModelConfig, 
			 training_config: WorldModelTrainingConfig,
			 env: MiniGridEnv, 
			 agent: Agent,
			 training_metrics_callback: TrainingCallback,
			 experiment_id=0 , device='cpu'):
		
		model = WorldModel(model_config)
		experience_buffer = collect_experiences(env, agent,training_config)

		if training_config.compute_proposed_class_weights:
			model.proposed_class_weights = compute_class_weights(experience_buffer=experience_buffer).to(device)
			print('Proposed Class Weights: ', model.proposed_class_weights, 'size:', model.proposed_class_weights.shape)

		train_dataloader = DataLoader(experience_buffer, batch_size=training_config.batch_size, shuffle=True)
		dataset_size = len(train_dataloader)

		param_list = list(model.parameters())
		# optimizer = torch.optim.Adam(param_list, lr=initial_learning_rate, weight_decay=1e-5)
		optimizer = torch.optim.Adam(param_list, lr=training_config.initial_learning_rate)
		learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=training_config.learning_rate_gamma)
		
		total_model_params = sum(p.numel() for p in model.parameters())
		total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print('Total Trainable Params {:,d}'.format(total_trainable_params))
		print('Total Params {:,d}'.format(total_model_params))

		# initialize training metrics tracker
		metrics_tracker = MetricTracker()

		cur_temperature = training_config.initial_temperature

		for epoch in range(training_config.epochs):

			train_pbar = tqdm.tqdm(train_dataloader)

			for batch_idx, sample_batch in enumerate(train_pbar):
				(episode, step, observation, action, next_observation, rewards, terminated, truncated) = sample_batch
				# convert environment sample data to tensors
				(observation_tensor, action_tensor, next_observation_tensor, rewards_tensor) = data_to_tensors(observation, action, next_observation, rewards, device=device)
				batch_inputs = (observation_tensor, action_tensor, next_observation_tensor, rewards_tensor)
				# update model
				loss_output, world_model_output = update(model, optimizer, training_config, batch_inputs, cur_temperature)
	

				if ((batch_idx % 10) == 0):
					train_pbar.set_description('Training, Epoch: [{}/{}] | Total loss: {:.7f} -- O_recon: {:.7f} --  O_kld: {:.7f} -- T_loss: {:.7f} -- R_loss: {:.7f}'\
								.format(epoch+1,
										training_config.epochs,
										loss_output.total_loss,
										loss_output.observation_recon_loss,
										loss_output.observation_kld,
										loss_output.transition_loss,
										loss_output.reward_loss))


				# calculate epoch-level eval recon accuracy metrics 
				correct_next_obs_recon_batch, correct_obs_recon_batch, correct_pred_obs_recon_batch	= training_reconstruction_accuracy(next_observation_tensor, observation_tensor, world_model_output)

				# epoch-level metrics
				metrics_tracker.track('training_loss', loss_output.total_loss.item(), epoch, batch_idx)
				metrics_tracker.track('observation_loss', loss_output.observation_loss.item(), epoch, batch_idx)
				metrics_tracker.track('transition_loss', loss_output.transition_loss.item(), epoch, batch_idx)
				metrics_tracker.track('reward_loss', loss_output.reward_loss.item(), epoch,batch_idx)
				metrics_tracker.track('next_obs_recon_accuracy', correct_next_obs_recon_batch, epoch, batch_idx)
				metrics_tracker.track('obs_recon_accuracy', correct_obs_recon_batch, epoch, batch_idx)
				metrics_tracker.track('pred_obs_recon_accuracy', correct_pred_obs_recon_batch, epoch, batch_idx)

			# print epoch level reconstruction accuracy report
			total_cell_count = (len(experience_buffer)*(observation_tensor.shape[1]*observation_tensor.shape[2]))
			print('NextObs acc: {:.2f}% -- Obs acc: {:.2f}% -- predObs acc: {:.2f}% \n'.format(\
				metrics_tracker.get_epoch_recon_accuracy('next_obs_recon_accuracy', epoch, total_cell_count), 
				metrics_tracker.get_epoch_recon_accuracy('obs_recon_accuracy', epoch, total_cell_count), 
				metrics_tracker.get_epoch_recon_accuracy('pred_obs_recon_accuracy', epoch, total_cell_count)))

			# incrementally anneal temperature and learning rate if enabled
			if ((epoch % 1) == 0) and (training_config.temp_anneal):
				cur_temperature = np.maximum(training_config.initial_temperature*np.exp(-training_config.temperature_anneal_rate*epoch), training_config.minimum_temperature)
				
			learning_rate_scheduler.step() # multiply learning rate by learning_rate_gamma
			print('\tupdated temperature: {:.3f}'.format(cur_temperature))
			print('\tcurrent learning rate: {:.3e}'.format(learning_rate_scheduler.get_last_lr()[0])) 

		training_metrics_callback('training_loss', metrics_tracker.get_epoch_average('training_loss'))
		training_metrics_callback('observation_loss', metrics_tracker.get_epoch_average('observation_loss'))
		training_metrics_callback('transition_loss', metrics_tracker.get_epoch_average('transition_loss'))
		training_metrics_callback('reward_loss', metrics_tracker.get_epoch_average('reward_loss'))

		# final model setup 
		model.temp = cur_temperature

		return model, optimizer

	
def update(model, optimizer, training_config, inputs, temp, device='cpu'):
	#TODO: need to account for dynamic loss weights like kld 
	''' runs a single learning step '''
	optimizer.zero_grad()

	param_list = list(model.parameters())

	(observation, action, next_observation, rewards) = inputs

	world_model_output = model(observation, next_observation, action, temp, gumbel_hard=model.gumbel_hard)

	# compute loss
	(o_recon_loss, pred_belief_loss, o_kld, r_loss) = world_model_loss(next_observation, observation, rewards, world_model_output,
											proposed_class_weights=model.proposed_class_weights, device=device)
	
	o_kld = o_kld * training_config.kl_loss_weight
	o_loss = o_recon_loss + o_kld
	t_loss = pred_belief_loss[1] + (pred_belief_loss[0]*training_config.pred_belief_loss_weight)
	loss = o_loss + t_loss + r_loss

	loss.backward()
	torch.nn.utils.clip_grad_norm_(param_list, training_config.grad_clip_norm, norm_type=2)
	optimizer.step()

	loss_output = model.LossOutput(loss, o_loss, t_loss, r_loss, o_recon_loss, o_kld)

	return loss_output, world_model_output

def load_model_from_checkpoint(model_id, optimizer, checkpoint_file, epoch, root_save_path='data/models', device='cpu', frozen=False):
	# get checkpoint
	checkpoint = torch.load(os.path.join(root_save_path, model_id, checkpoint_file))
	# instantiate model based on config
	config = WorldModelConfig(**checkpoint['world_model_config'])
	model = WorldModel(config)
	# load all the weights
	model.load_state_dict(checkpoint['world_model_state_dict']).to(device)
	model.observation_model.load_state_dict(checkpoint['observation_model_state_dict']).to(device)
	model.transition_model.load_state_dict(checkpoint['transition_model_state_dict']).to(device)
	model.reward_model.load_state_dict(checkpoint['reward_model_state_dict']).to(device)
	# load optimizer
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	if frozen:
			model.freeze_weights()
	return model, optimizer, epoch

def save_model_checkpoint(model, optimizer, epoch, root_save_path='data/models'):
	'''
	Save model weights
	'''

	world_model_dict = {
		'observation_model_state_dict' : model.observation_model.state_dict(),
		'transition_model_state_dict': model.transition_model.state_dict(),
		'reward_model_state_dict': model.reward_model.state_dict(),
		'world_model_state_dict': model.state_dict(),
		'world_model_config': asdict(model.config),
		'world_model_proposed_class_weights': model.proposed_class_weights,
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch
	}

	path_to_model = os.path.join(root_save_path, model.model_id)
	if not os.path.isdir(path_to_model):
			os.makedirs(path_to_model)

	datetime_ = datetime.datetime.now().strftime("%Y%m%d")
	seconds_ = int(time.time()) 
	model_filename = 'checkpoint-trained_models_{}-{}.pth'.format(datetime_,seconds_)
	model_filepath = os.path.join(path_to_model, model_filename)
	print('Saving Model: \n\t{}'.format(model_filepath))
	torch.save(world_model_dict, model_filepath)

def sample_from_latent(model, action_list, 
					   prior_type: Literal['uniform', 'prior'], 
					   device='cpu', convert_output_to_array=True):

	SampleOutputs = namedtuple('SampleOutputs', ['latent', 'next_latent', 
											  	 'observation','next_observation',
												 'action_list'])
	cur_temp = model.temp
	print('using temperature', cur_temp)

	# sample a batch random states (B, N, K): batch, num_distributions, num_categories
	num_samples = len(action_list)
	B = num_samples
	N =  model.config.num_categorical_distributions
	K = model.config.categorical_dim
	latent_shape = (B, N ,K)
		
	random_actions_tensor = torch.tensor(action_list, device=device, dtype=torch.int)

	if prior_type == 'uniform':
		# sample using uniform
		states = uniform_latent_sampler(latent_shape, 
										temperature=cur_temp, 
										hard=False).view(-1, N*K).view(-1, N*K, 1, 1).to(device)	
	elif prior_type == 'learned':
		# use learn, ensure no gradients are accumulated
		with torch.no_grad():
			prior_logits = model.observation_model.prior(torch.zeros(B, N, device=device)).view(-1, N, K)
			states = learned_prior_sampler(prior_logits, 
								  temperature=cur_temp, 
								  hard=False).view(-1, N*K).view(-1, N*K, 1, 1).to(device)
	else:
		raise NotImplementedError('prior_type {} is not currently supported.'.format(prior_type))
	
	with torch.no_grad():
		observation_logits = model.observation_model.decode(states)
		observation = torch.max(observation_logits, dim=1, keepdim=True).indices.squeeze()
		latent = states.reshape(num_samples, K, N)

		transition_model_output = model.transition_model(states, random_actions_tensor, cur_temp)
		action_embed = transition_model_output.action_embed
		next_states = transition_model_output.pred_next_latent_belief.squeeze().view(-1, N*K, 1, 1)
		next_latent = next_states.reshape(num_samples, K, N)

		next_obs_logits = model.observation_model.decode(next_states)
		next_observation = torch.max(next_obs_logits, dim=1, keepdim=True).indices.squeeze()

	if convert_output_to_array: 
		latent = latent.cpu().numpy()
		next_latent = next_latent.cpu().numpy()
		observation = observation.cpu().numpy()
		next_observation = next_observation.cpu().numpy()

	return SampleOutputs(latent, next_latent, observation, next_observation, action_list)