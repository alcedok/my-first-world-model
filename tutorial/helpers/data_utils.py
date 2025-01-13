import tqdm 
import numpy as np
from collections import namedtuple

from minigrid.core.actions import Actions

from confs.definitions import WorldModelTrainingConfig

def collect_experiences(env, agent, config: WorldModelTrainingConfig):

	num_rollouts = config.warm_up_rollouts

	pbar = tqdm.tqdm(range(num_rollouts), desc='Collecting experiences')

	total_steps = 0
	episode_steps = []
	experience_buffer = []
	goal_reached = 0

	DataSample = namedtuple('DataSample', ['episode', 'step', 'observation', 'action', 
										 	'next_observation', 'rewards', 'terminated', 'truncated'])
	for episode in pbar:
		step_count = 0
		observation, _ = env.reset()
		agent.reset(observation)
		done = False
		step = 0
		while not done:
			step += 1
			action = agent.act(observation)
			next_observation, rewards, terminated, truncated, _ = env.step(action)
			step_count += 1
			sample = DataSample(episode, step, observation, action, next_observation, rewards, terminated, truncated)
			experience_buffer.append(sample)
			observation = next_observation
			done = truncated or terminated
			if terminated:
				goal_reached+=1
			if done:
				break

		episode_steps.append(step_count)
		total_steps += step_count

	print('Total episodes collected: {:,}'.format(num_rollouts))
	print('Total steps collected: {:,}'.format(total_steps))
	print('Avg. steps/episode: {:,}'.format(np.mean(episode_steps)))
	print('Percent goal reached: {:.1f}%'.format(100*(goal_reached/num_rollouts)))
	env.close()
	return experience_buffer