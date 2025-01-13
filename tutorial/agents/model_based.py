import torch 

from models.utils import data_to_tensors, uniform_latent_sampler
from models.dynaq import update as policy_update
from models.world_model import update as model_update
from agents.abstract import Agent
from helpers.metrics_utils import TrainingCallback
from confs.definitions import ModelBasedTrainingConfig

class ModelBasedAgent(Agent):
	def __init__(self,
			  config: ModelBasedTrainingConfig, 
			  policy,
			  model,
			  model_optimizer,
			  policy_metrics_callback: TrainingCallback, 
			  model_metrics_callback: TrainingCallback,
			  agent_name='Model Based Agent',
			  device='cpu'):
		
		# world model specific configs
		self.wm_config = config.world_model
		self.wm_training_config = config.world_model_training
		self.dynaq_config = config.dynaq

		self.valid_actions = config.dynaq.valid_actions
		self.policy_metrics_callback = policy_metrics_callback
		self.model_metrics_callback = model_metrics_callback
		self.agent_name = agent_name
		self.device = device

		self.policy = policy
		self.model = model
		self.model_optimizer = model_optimizer

	def mental_simulation(self):
		''' like plan ahead but unguided exploration '''
		temp = self.model.temp
		num_simulations = self.dynaq_config.num_simulations
		
		# sample a batch random states (B, N, K): batch, num_distributions, num_categories
		B = num_simulations
		N =  self.wm_config.num_categorical_distributions
		K = self.wm_config.categorical_dim
		latent_shape = (B, N ,K)

		states = uniform_latent_sampler(latent_shape, temperature=temp, hard=False).view(-1, N*K)
		# collect all next steps for each valid action
		valid_actions_tensor = self.policy.valid_actions_tensor

		# repeat valid_actions to the number of simulations (i.e test all valid actions given a sample)
		# hence we need to repeat the valid_vectors, but state is repeat_interleaved?? Maybe not
		# valid actions are of shape: (M,)
		# batch state has shape: (B, K*N)
		# desired shape for both before computing transition: ( B*M, K*N )
		M = valid_actions_tensor.shape[0]
		repeat_actions = valid_actions_tensor.repeat(B) # (B*M, )
		repeat_states = states.repeat_interleave(M, dim=0) # (B*M, K*N)

		transition_model_output = self.model.transition_model(repeat_states, repeat_actions, temp)
		action_embed = transition_model_output.action_embed
		next_states = transition_model_output.pred_next_latent_belief.squeeze()

		reward_model_output = self.model.reward_model(repeat_states)
		rewards = reward_model_output.reward

		terminated = False
		inputs = (repeat_states, next_states, action_embed, rewards, terminated)
		policy_update(self.policy, self.policy.optimizer, inputs, embed_actions=False)

	def reset(self, observation=None):
		return 
	
	def act(self, observation=None):
		observation_tensor = torch.tensor(observation['observation']).to(torch.int).to(self.device).unsqueeze(0)
		_, _, state, _ = self.model.observation_model.get_belief(observation_tensor)
		state = state.squeeze().unsqueeze(0)
		return self.policy.choose_action(state)
	
	def update_policy(self, observation, action, reward, next_observation, terminated):
		(observation_tensor, action_tensor, next_observation_tensor, rewards_tensor) = data_to_tensors(observation, action, next_observation, reward, device=self.device)
		# include a batch dimension
		observation_tensor = observation_tensor.unsqueeze(0)
		next_observation_tensor = next_observation_tensor.unsqueeze(0)
		action_tensor = action_tensor.unsqueeze(0)
		rewards_tensor = rewards_tensor.unsqueeze(0)
		# get action embedding, make sure no gradients are accumulated
		with torch.no_grad():
			_, _, state, _ = self.model.observation_model.get_belief(observation_tensor)
			_, _, next_state, _ = self.model.observation_model.get_belief(next_observation_tensor)
			# action_embed = self.policy.ActionEmbedding(action_tensor).unsqueeze(0)

		inputs = (state.squeeze(-1,-2), next_state.squeeze(-1,-2), action_tensor, rewards_tensor, terminated)
		loss_output, policy_output = policy_update(self.policy, self.policy.optimizer, inputs, embed_actions=True)
		# self.policy_metrics_callback('training_loss', loss_output.total_loss.item())
	
	def update_model(self, observation, action, reward, next_observation, terminated):
		(observation_tensor, action_tensor, next_observation_tensor, rewards_tensor) = data_to_tensors(observation, action, next_observation, reward, device=self.device)
		temp = self.model.temp
		observation_tensor = observation_tensor.unsqueeze(0)
		next_observation_tensor = next_observation_tensor.unsqueeze(0)
		action_tensor = action_tensor.unsqueeze(0)
		rewards_tensor = rewards_tensor.unsqueeze(0)
		inputs = (observation_tensor, action_tensor, next_observation_tensor, rewards_tensor)
		loss_output, world_model_output = model_update(self.model, self.model_optimizer, self.wm_training_config, inputs, temp)
		# self.model_metrics_callback('training_loss', loss_output.total_loss.item())