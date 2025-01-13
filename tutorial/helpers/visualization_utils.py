
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

from helpers.metrics_utils import Metrics

def plot_latent(ax, latent_data):
	''' plot latent data using either imshow or pcolormesh. '''
	if latent_data.ndim == 3 and latent_data.shape[2] == 1:
		latent_data = latent_data.squeeze(axis=2)

	if latent_data.ndim == 1 or (latent_data.ndim == 2 and 1 in latent_data.shape):
		if latent_data.ndim == 2:
			latent_data = latent_data.flatten()

		latent_data = latent_data.reshape(1, -1).T
		ax.imshow(latent_data, cmap='viridis', aspect='auto', interpolation='nearest')
		aspect_ratio = (latent_data.shape[0] / latent_data.shape[1] ) / 5000 # dynamically change aspect ratio to make it a column
		ax.set_aspect(aspect_ratio)

	elif latent_data.ndim == 2:
		ax.imshow(latent_data)
		
	else:
		print('latent data has unsupported dimensions')

def plot_samples(env, samples, title=''):
	''' plot grid of: latent, obs, next_latent_obs '''
	(latents, next_latents, obs, next_obs, actions) = samples

	actions_str = [a.name for a in actions]
	
	obs = env.unwrapped.partial_to_full_obs(obs)
	next_obs = env.unwrapped.partial_to_full_obs(next_obs)

	num_rows = obs.shape[0]

	fig = plt.figure(figsize=(10, 2*num_rows))
	gs = gridspec.GridSpec(num_rows, 5, width_ratios=[1, 1, 0.3, 1, 1], height_ratios=[1]*num_rows)

	for row in range(num_rows):
		ax_pos_latents = fig.add_subplot(gs[row, 0])
		ax_pos_obs = fig.add_subplot(gs[row, 1])
		ax_pos_action_text = fig.add_subplot(gs[row, 2])
		ax_pos_next_latents = fig.add_subplot(gs[row, 3])
		ax_pos_next_obs = fig.add_subplot(gs[row, 4])

		# ax_pos_latents.imshow(latents[row])
		plot_latent(ax_pos_latents, latents[row])
		ax_pos_obs.imshow(env.unwrapped.obs_to_image(obs[row]))
		ax_pos_action_text.text(0.5, 0.5, actions_str[row], ha='center', va='center', fontsize=12)
		for spine in ax_pos_action_text.spines.values():
			spine.set_visible(False)
		# ax_pos_next_latents.imshow(next_latents[row])
		plot_latent(ax_pos_next_latents, next_latents[row])
		ax_pos_next_obs.imshow(env.unwrapped.obs_to_image(next_obs[row])) 

	col_titles = ['Latent Code', 'Sampled Obs', 'Action', 'Next Latent Code', 'Next Obs Recon']
	for ax, col in zip(fig.axes[:5], col_titles):
		ax.set_title(col)
	
	fig.suptitle(title, fontsize=16)
	plt.subplots_adjust(left=None, bottom=None, right=None, wspace=0.05, hspace=0.05)
	plt.setp(fig.axes, xticks=[], yticks=[]) # set on all axes in figure

	plt.show()

def training_plots(metrics: Metrics, metrics_keys=None, ylabel='Loss', xlabel='Epochs'):
	''' training curves '''
	if metrics_keys is None:
		# infer from inputs
		metrics_keys = list(metrics.keys())

	num_plots = len(metrics_keys)

	cmap = cm.get_cmap('tab10')

	# check if single subplot needed
	if num_plots == 1:
		fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figsize as needed
		axes = [ax]  # Wrap in a list for consistent iteration
	else:
		fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))

	for i, ax in enumerate(axes):
		metric_name = metrics_keys[i]
		title = metric_name.replace('_', ' ').capitalize()
		data = metrics[metric_name]

		x_vals = np.arange(1, len(data) + 1)
		ax.plot(x_vals, data, label=title, marker='o', color=cmap(i))
		ax.set_title(title)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.legend()
		ax.set_xticks(range(1,len(data)+1))
		ax.grid(True)
		# ax.set_yscale('log')

	plt.tight_layout()
	plt.show()

def smooth_rewards(rewards, window=10):
	''' smoothes rewards using a rolling average '''
	if len(rewards) < window:
		return rewards
	series = pd.Series(rewards)
	smoothed_series = series.rolling(window=window, min_periods=1).mean()
	return list(smoothed_series)

def plot_episode_rewards(rewards, title='Episode Rewards', window=20):
	''' episode rewards with smoothing functionality '''
	cmap = cm.get_cmap('tab10')

	plt.figure(figsize=(10, 4))
	# plt.plot(rewards, label='raw rewards', alpha=0.7)  # alpha for transparency
	
	smoothed_rewards = smooth_rewards(rewards, window=window)
	plt.plot(smoothed_rewards, label='smoothed rewards'.format(window), linewidth=2, color=cmap(0))

	window_std = pd.Series(rewards).rolling(window=window, min_periods=1).std()
	plt.fill_between(range(len(rewards)), 
				  list(np.array(smoothed_rewards) - np.array(window_std)), 
				  list(np.array(smoothed_rewards) + np.array(window_std)), 
				  alpha=0.3, label='standard deviation', color=cmap(0))
	
	plt.xlabel('episode')
	plt.ylabel('reward')
	plt.title(title)
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.show()