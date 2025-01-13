from __future__ import annotations

import numpy as np
import random 
import matplotlib.pylab as plt
from matplotlib.colors import Normalize
import pygame
import math 

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Door
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.actions import Actions
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, COLORS, DIR_TO_VEC

from minigrid.utils.rendering import (
	fill_coords,
	point_in_rect,
	point_in_triangle,
	rotate_fn,
)

from environments.env_models import EnvModel, TransitionModel, ObservationModel
from environments.constants import Directions, DirectionsXY

class DoorCrossing(MiniGridEnv):
	'''
	## Description

	- Reach the cell with the green goal by crossing the automatic door.
	- The door is there to prevent seeing over to the next room, a known issue of Minigrid: https://github.com/Farama-Foundation/Minigrid/issues/101
	- (optional + computational overhead) 
		- A transition (T) and Observation (O) matrix can be generated at each reset
		- T and O can be generated with optional non-determinism, i.e action/transition and sensor noise

	## Mission Space
	- 'find the opening and get to the green goal square'

	## Action Space

	| Num | Name         | Action       |
	|-----|--------------|--------------|
	| 0   | left         | Turn left    |
	| 1   | right        | Turn right   |
	| 2   | forward      | Move forward |
	| 3   | pickup       | Unused       |
	| 4   | drop         | Unused       |
	| 5   | toggle       | Unused       |
	| 6   | done         | Unused       |

	## Observation Encoding

	- Each tile is encoded as a 3 dimensional tuple:
		`(OBJECT_IDX, COLOR_IDX, STATE)`
	- `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
		[minigrid/core/constants.py](minigrid/core/constants.py)
	- `STATE` refers to the door state with 0=open, 1=closed and 2=locked

	## Rewards

	A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

	## Termination

	The episode ends if any one of the following conditions is met:

	1. The agent reaches the goal.
	2. Timeout (see `max_steps`).

	'''

	def __init__(
		self,
		valid_actions,
		width=5,
		height=5,
		see_through_walls=False,
		randomize=False,
		randomize_agent=False,
		max_steps = None,
		agent_view_size = 3,
		curriculum_learning_prob = 0.7,
		compute_env_model = False,
		render_mode = 'rgb_array',
		nondeterministic=False, 
		robot_model_data:dict=None,
		**kwargs,
	):
		if nondeterministic and robot_model_data is None:
			raise ValueError('When nondeterministic flag is enabled, a transition/observation robot_model_data dict must be included')
		
		self.width = width
		self.height = height
		self.agent_view_size = agent_view_size
		self.goal_position = None
		self.see_through_walls = see_through_walls
		self.randomize = randomize
		self.randomize_agent = randomize_agent
		self.render_mode = render_mode
		self.valid_actions = valid_actions
		self.num_valid_actions = len(self.valid_actions)
		self.curriculum_learning_prob = curriculum_learning_prob # range [0,1], probability of being in goal region when randomize

		# if enabled, compute Transition and Observation models at each reset()
		self.compute_env_model = compute_env_model  
		self.robot_model_data = robot_model_data
		self.env_model = None
		
		mission_space = MissionSpace(mission_func=self._gen_mission)
				
		self.agent_dir_options = list(Directions)
	
		# environment constants
		# minigrid uses (col,row) for grid positions 
		self.door = Door(color='blue', is_open=False, is_locked=False)

		# valid regions when random sampling placement of objects
		self.init_region = [(j,i) for j in range(1,self.width//2) for i in range(1,self.height-1)]
		self.door_region = [(self.width//2,i) for i in range(1,self.height-1)]
		self.goal_region = [(j,i) for j in range(self.width//2+1, self.width-1) for i in range(1,self.height-1)]
		
		# static object positions
		# agent: top-left corner (col,row), facing down
		self.agent_init_pos = (1, 1)
		self.ego_agent_pos = (self.agent_view_size // 2, self.agent_view_size - 1)
		self.agent_init_dir = Directions.down
		# goal: bottom-right corner, (col,row)
		self.goal_init_position = (self.width - 2, self.height - 2)
		# in the middle, (col,row)
		self.door_init_position = (self.width//2, self.height//2)
		# dividing-wall: vertical, in the middle
		self.wall_position = self.width//2
		 
		if max_steps is None:
			max_steps = 500

		super().__init__(
			mission_space=mission_space,
			width=self.width,
			height=self.height,
			see_through_walls=self.see_through_walls,
			max_steps=max_steps,
			agent_view_size=self.agent_view_size,
			render_mode = render_mode,
			**kwargs,
		)
		
		self.action_space.n = self.num_valid_actions
		
		assert self.num_valid_actions==self.action_space.n, \
			'Number of valid_actions {} != Size of action_space {}'.format(self.num_valid_actions, self.action_space.n)


	@staticmethod
	def _gen_mission():
		return 'find the opening and get to the green goal square'
	
	def check_valid_agent_pos(self):
		start_cell = self.grid.get(*self.agent_pos)
		assert start_cell is None or start_cell.can_overlap(), \
			'position {} overlaps with object: {}'.format(self.agent_pos, start_cell)

	def set_goal(self):
		self.put_obj(Goal(), *self.goal_position)

	def set_wall(self):
		# place vertical wall along the middle of the gridworld    
		self.grid.vert_wall(self.wall_position,0) 
	
	def set_door(self):
		self.door.is_open=False
		self.put_obj(self.door, *self.door_position)

	def door_controller(self):
		front_cell = self.grid.get(*self.front_pos)
		curr_cell = self.grid.get(*self.agent_pos)
		
		# if the door is in front of agent then open automatically
		# also keep the door open while the agent is going through
		# if the agent is initialized on top of the door, set it to open 
		if (front_cell and front_cell.type == 'door') or (curr_cell and curr_cell.type == 'door'):
			self.door.is_open = True
		else:
			self.door.is_open =False
	
	def _gen_grid(self, width, height):
		# ensure gridwolrd has odd size
		assert width % 2 == 1 and height % 2 == 1 , \
			'grid dimensions must be odd for proper wall placement.'

		# create an empty grid
		self.grid = Grid(width, height)

		# generate the surrounding walls
		self.grid.wall_rect(0, 0, width, height)

		# randomize world objects?
		if self.randomize:
			self.goal_position = random.choice(self.goal_region)
			self.door_position = random.choice(self.door_region)
		else:
			self.goal_position = self.goal_init_position
			self.door_position = self.door_init_position
			
		# randomize agent?
		if self.randomize_agent:
			if random.random() < self.curriculum_learning_prob: # prioritize regions near the goal by prob
				goal_region_set =  {p for p in self.goal_region}
				goal_region_set.discard(self.goal_position)
				self.agent_pos = random.choice(list(goal_region_set))
			else:
				region_set = {p for p in self.init_region+self.goal_region}
				region_set.discard(self.goal_position)
				self.agent_pos = random.choice(list(region_set))
			self.agent_dir = random.choice(self.agent_dir_options)
		else: 
			self.agent_pos = self.agent_init_pos
			self.agent_dir = self.agent_init_dir

		# set objects on the grid
		self.set_wall()
		self.set_door()
		self.set_goal()

		self.door_controller()
		self.check_valid_agent_pos()

		if self.compute_env_model:
			self.env_model = self.generate_env_model()

		self.mission = ('find the opening and get to the green goal square')
	
	def generate_env_model(self):
		# compute models for new grid
		env_model = EnvModel(self.grid, self.agent_view_size, self.robot_model_data, self.valid_actions)
		return env_model

	@property
	def transition_matrix(self):
		return self.env_model.transition_matrix 
	
	@property
	def observation_matrix(self):
		return self.env_model.observation_matrix 
	
	def set_curriculum(self, value):
		self.curriculum_learning_prob = value
	
	def reset(self, seed=None, options=None):
		# call to parent MiniGridEnv's reset() for grid generation and default init
		observations, info = super().reset(seed=None, options=options)
		return observations, info 

	def step(self, action):
		# invalid action
		if action not in self.valid_actions:
			raise ValueError('Invalid action {}; Must be one of {}'.format(action, self.valid_actions))		
		# update the agent's position/direction
		obs, reward, terminated, truncated, info = super().step(action)
		self.door_controller()
		return obs, reward, terminated, truncated, info

	'''
	Rendering methods
	'''
	def render(self):
		img = super().render()
		return img
	
	def get_array_repr(self, with_agent=True):
		grid_array = self.unwrapped.grid.encode()[:,:,0]
		grid_array[self.agent_pos[0], self.agent_pos[1]] = OBJECT_TO_IDX['agent']
		return grid_array.T
	
	def show_render(self):
		plt.imshow(self.render())
		plt.axis('off')
		return 
	
	def obs_to_image(self, obs):
		grid, _ = self.grid.decode(obs)
		image = grid.render(tile_size=self.tile_size, 
						agent_pos=self.ego_agent_pos, 
						agent_dir=3)
		return image
	
	def partial_to_full_obs(self, obs):
		'''
		must be in batches of observations [B,N,N]
		this is very hacky, but just for the sake of keeping things simple
		'''

		## color dimension
		color_dim = np.ones_like(obs)
		env_obj_colors = {'wall':'grey', 'unseen':'grey', 'door':'blue' }

		for obj_name, obj_color in env_obj_colors.items():
			obj_id = OBJECT_TO_IDX[obj_name]
			color_id = COLOR_TO_IDX[obj_color]
			color_dim[obs==obj_id] = color_id
		
		## state dimension
		state_dim = np.zeros_like(obs)
		# where is the door
		door_positions = np.argwhere(obs == OBJECT_TO_IDX['door'])
		if len(door_positions)!=0:
			for (obs_idx, row, col) in door_positions:
				cells_to_keep_door_open = {
					(self.ego_agent_pos[0], self.ego_agent_pos[1]-1),
					(self.ego_agent_pos[0], self.ego_agent_pos[1])
					} #Minigrid uses (col, row)
				if (row,col) in cells_to_keep_door_open:
					# door_pos
					state_dim[obs_idx,row,col] =STATE_TO_IDX['open']
				else: 
					state_dim[obs_idx,row,col] =STATE_TO_IDX['closed']

		# combine dims
		full_dim = np.stack([obs,color_dim,state_dim],axis=-1)
		return full_dim
	
	def render_state_values(self, state_values: dict, window_size=1024):
		img = self.render()
		canvas = pygame.surfarray.make_surface(img)

		# color map
		cmap = plt.cm.get_cmap('viridis') 
		min_values = np.min(state_values)
		max_values = np.max(state_values)
		norm = Normalize(vmin=min_values, vmax=max_values)

		# get corresponding (r,g,b)
		values_to_rgba = cmap(norm(state_values))
		values_to_rgb = (values_to_rgba[:,:3]*255).astype(int)

		canvas.fill((0, 0, 0))
		pix_square_size = (window_size / self.tile_size)  # size of a single grid square in pixels

		triangle_points = [(0.12, 0.19), (0.87, 0.50), (0.12, 0.81),] # from Minigrid point_in_triangle()
		triangle_points = [(p[0]*self.tile_size, p[1]*self.tile_size) for p in triangle_points] # scaled
		triangle_center = (pix_square_size/2, pix_square_size/2)

		# draw values 
		for cell in self.env_model.free_cells_set:
			x,y = cell
			state_value_dir = [
						(state_values[self.env_model.transition_model.rlookup[(x,y,o)]], 
							self.env_model.transition_model.rlookup[(x,y,o)],
								o) 
						for o in list(Directions)]
			(value, state_id, orientation) = max(state_value_dir, key=lambda item: item[0])
			cell_rgb = tuple(values_to_rgb[state_id].tolist())
			offset_x = pix_square_size * y
			offset_y = pix_square_size * x
			rotation_angle = vector_to_angle(DIR_TO_VEC[orientation.value])
			rotated_triangle = rotate_points(triangle_points, rotation_angle, triangle_center)
			translated_triangle = [(x + offset_x, y + offset_y) for x, y in rotated_triangle]
			pygame.draw.polygon(canvas, cell_rgb, translated_triangle)
			
		# draw entities, except doors
		for j in range(0, self.height):
			for i in range(0, self.width):
				cell = self.grid.get(i, j)
				if (cell is not None) and (cell.type != 'door'):
						pygame.draw.rect(
							surface=canvas,
							color=COLORS[cell.color],
							rect=pygame.Rect(
								pix_square_size * np.array([j,i]),
								(pix_square_size, pix_square_size),
							),
						)

		return np.array(pygame.surfarray.pixels3d(canvas))
	
	def plot_state_values(self, input):
		''' 
		draw state-value pairs.
		if input is a Q_lookup then only draw the max value for each action
		'''
		if isinstance(input, dict):
			# input comes from a Q(s,a) lookup
			# we generate subplots for each action
			# all_q_value_action_map = {}
			images = {}
			for action in self.valid_actions:
				all_q_value_action = [(value, state_id, action_i) 
													for (state_id, action_i), value 
													in input.items() if action_i == action]
				sorted_q_values_by_state = sorted(all_q_value_action , key=lambda item: item[1])
				sorted_q_values = [value for (value, state_id, action_i) in sorted_q_values_by_state] + [0,0] # we include impossible goal states
				images[action.name] = self.render_state_values(sorted_q_values)

			num_plots = len(self.valid_actions)
			fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
			images_to_plot = {i: (action, img) for i, (action, img) in enumerate(images.items())}
			for i, ax in enumerate(axes):
				action_i, img_i = images_to_plot[i]
				ax.imshow(img_i)
				ax.set_title(action_i)		
			plt.setp(fig.axes, xticks=[], yticks=[]) # remove ticks
			
		else:
			# input comes from Value Iteration-like result: an array of values for each state_di
			img = self.render_state_values(input)
			plt.imshow(img)
			plt.colorbar()
			plt.axis('off')

def rotate_points(points, angle_degrees, center):
	''' Rotates a list of points around a center. '''
	angle_radians = math.radians(angle_degrees)
	cos_theta = math.cos(angle_radians)
	sin_theta = math.sin(angle_radians)

	rotated_points = []
	for x, y in points:
		x -= center[0]
		y -= center[1]

		rotated_x = x * cos_theta - y * sin_theta
		rotated_y = x * sin_theta + y * cos_theta

		rotated_x += center[0]
		rotated_y += center[1]
		rotated_points.append((rotated_x, rotated_y))
	return rotated_points

def vector_to_angle(vector):
	'''Converts a 2D vector to an angle in degrees (0-360).'''
	x, y = vector
	angle = math.degrees(math.atan2(x, y))
	return (angle + 360) % 360  # ensure angle is in range 0-360
