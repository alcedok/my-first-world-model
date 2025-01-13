
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from itertools import product
from typing import Tuple

from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.actions import Actions

from environments.constants import Directions, DirectionsXY

class RewardModel:
	def __init__(self, **kwargs):
		''' reward lookup, where R[state_id] returns a reward'''
		for key, value in kwargs.items():
			setattr(self, key, value)

		self.R = self.generate()
	
	def get_summary(self):
		print('Reward Model Summary:')
		print('\tnumber of reward states: {}'.format(len(self.R.keys())))

	def generate(self):
		R = {}
		for x, y in self.free_cells:
			cell_value = self.grid[x,y]
			if IDX_TO_OBJECT[cell_value] in self.reward_cells:
				for orientation in list(Directions):
					state_id = self.transition_model.rlookup[(x, y, orientation)]
					R[state_id] = 1.0
		return R
	
	def __getitem__(self, state_id: int):
		''' if state_id not in lookup default to returning 0.0 '''
		if not isinstance(state_id, int):
			raise TypeError('state_id must be int, received {}'.format(type(state_id).__name__))
		if state_id not in self.R.keys():
			return 0.0 
		return self.R[state_id]
	
class TransitionModel:
	def __init__(self, **kwargs):
		for key, value in kwargs.items():
					setattr(self, key, value)

		self.states = self.generate_states_and_lookups()
		self.lookup = {state_id: (x,y,o) for state_id, (x,y,o) in enumerate(self.states)}
		self.rlookup = {(x,y,o): state_id for state_id, (x,y,o) in enumerate(self.states)}
		self.T = self.compute()

	def get_summary(self):
		print('Transition Model Summary:')
		print('\tT(s,a,s\'): {:,} x {:,} x {:,}'.format(self.T.shape[0], self.T.shape[1], self.T.shape[2]))

	def generate_states_and_lookups(self):
		orientations = list(Directions)
		states = []
		for (x, y) in self.free_cells:
			for orientation in orientations:
				states.append((x, y, orientation))

		assert len(states) == self.num_states, 'number of states generated {} do not match the expected {}'.format(len(states), self.num_states)
		return states
	
	def init(self):
		return np.zeros((self.num_states, self.num_states))

	def compute(self):
		'''
		Compute the state transition probability tensor T
		where T[s, a, s'] = P(X_t+1 = s' | X_t = s, A_t = a)

		T is represented as a 3D numpy array with shape:
			(num_states, num_actions, num_states)
		- num_states: Total number of states (NxMxD, considering grid size and orientations)
		- num_actions: Total number of actions (e.g., forward, left, right)

		The transition probabilities account for intended actions and noise.
		'''

		num_states = len(self.states)
		T = np.zeros((num_states, self.num_actions, num_states))  # Initialize 3D tensor

		# Iterate over all states
		for cur_state_id, (x, y, o) in self.lookup.items():
			# Compute transitions for each action
			for action_idx, action in enumerate(self.valid_actions):
				action_name = action.name

				# Initialize self-loop for invalid transitions
				T[cur_state_id, action_idx, cur_state_id] += self.robot_model_data['transitions'][action_name]['intended']

				left_orientation = (o.value - 1) % len(Directions)
				right_orientation = (o.value + 1) % len(Directions)

				left_state = (x, y, Directions(left_orientation))
				right_state = (x, y, Directions(right_orientation))

				left_state_id = self.rlookup[left_state]
				right_state_id = self.rlookup[right_state]

				if action_name == 'forward':
					# Forward movement
					dx, dy = DirectionsXY[o.name].value
					next_x, next_y = x + dx, y + dy

					# Check if forward move is valid
					if (0 <= next_x < self.grid.shape[0]) and (0 <= next_y < self.grid.shape[1]) and ((next_x, next_y) in self.free_cells_set):
						next_state = (next_x, next_y, o)
						next_state_id = self.rlookup[next_state]
						T[cur_state_id, action_idx, next_state_id] += self.robot_model_data['transitions']['forward']['intended']
					else:
						# If forward move is invalid, it's a self-loop
						T[cur_state_id, action_idx, cur_state_id] += self.robot_model_data['transitions']['forward']['intended']

			
					T[cur_state_id, action_idx, left_state_id] += self.robot_model_data['transitions']['forward']['slip_left']
					T[cur_state_id, action_idx, right_state_id] += self.robot_model_data['transitions']['forward']['slip_right']

				elif action_name == 'left':
					# Turning left
					left_orientation = (o.value - 1) % len(Directions)
					left_state = (x, y, Directions(left_orientation))
					left_state_id = self.rlookup[left_state]

					T[cur_state_id, action_idx, left_state_id] += self.robot_model_data['transitions']['left']['intended']
					T[cur_state_id, action_idx, cur_state_id] += self.robot_model_data['transitions']['left']['slip_forward']
					T[cur_state_id, action_idx, right_state_id] += self.robot_model_data['transitions']['left']['slip_right']

				elif action_name == 'right':
					# Turning right
					right_orientation = (o.value + 1) % len(Directions)
					right_state = (x, y, Directions(right_orientation))
					right_state_id = self.rlookup[right_state]

					T[cur_state_id, action_idx, right_state_id] += self.robot_model_data['transitions']['right']['intended']
					T[cur_state_id, action_idx, cur_state_id] += self.robot_model_data['transitions']['right']['slip_forward']
					T[cur_state_id, action_idx, left_state_id] += self.robot_model_data['transitions']['right']['slip_left']

			# Normalize the transition probabilities for each action
			for a in range(self.num_actions):
				row_sum = T[cur_state_id, a, :].sum()
				if row_sum > 0:
					T[cur_state_id, a, :] /= row_sum

		# Check that the transition tensor is valid
		for s in range(num_states):
			for a in range(self.num_actions):
				row_sum = T[s, a, :].sum()
				if not np.isclose(row_sum, 1.0):
					raise ValueError('Transition probabilities for state {}, action {} do not sum to 1.0: {}'.format(s, a, row_sum))

		assert np.all(T >= 0), 'T contains negative values'

		return T


class ObservationModel:
	def __init__(self, states, **kwargs):
		''''
		rlookup expects obs_id to be a tuple of flatten observations
		'''
		for key, value in kwargs.items():
					setattr(self, key, value)

		self.states = states
		# self.lookup = {obs.flatten(): obs_id for obs_id, obs in enumerate(self.all_possible_observations)}
		self.lookup = MonotonicCounter()
		self.O = self.compute()
		self.rlookup = {obs_id:obs for obs_id, obs in self.lookup.items()}
		self.num_actual_obs = len(self.lookup.keys())

	def get_summary(self):
		print('Observation Model Summary:')
		print('\tO (upper-bound): {:,} x {:,}'.format(self.O.shape[0], self.O.shape[1]))
		print('\tO (actual): {:,} x {:,}'.format(self.transition_model.T.shape[0], self.num_actual_obs))

	def init(self):
		''' we use a sparse matrix for efficient data storage '''
		return csr_matrix((self.num_states, self.num_observations_upper_bound))
	
	def compute_all_possible_observations(self):
		''' May take a long time! N^K operations '''
		# generate all possible combinations of observations
		# we convert the square observations to a flat vector
		#  i.e [(0,1,1,2,...),(0,1,1,8...)...] all the way to {num_entities}^{agent_view_sieze}
		return list(product(self.entities_int_list, repeat=self.flat_observations_dim))

	def compute(self) -> csr_matrix:
		'''
		Compute the observation probability matrix O
			where O[i,o] = Prob( O_t = o | X_t = i )
		
		O is represented as a 2D numpy array where [i,o] corresponds to the 
			probability of observing `o` given the agent is in state `i` 
			O has shape (NxN) x (Z); (row x col)
			where N is the state size. 
			where Z is the agent_view_size^{2} 
			The upper bound is (num_world_entities)^{Z} 
			for example, 
				in the case of 7x9 grid and 4 orientations, 
				with  and agent_view_size = 3, and num_world_entities=4,
				the upper bound shape for O will be (7x9x4) x (4^(3x3))
					i.e (252 x 262,144) ... yea, pretty big

		
		The observation probability Prob( o | i ) may be deterministic (with no sensor noise) 
			and could have aliasing.
		Aliasing is when the same observation may be possible from different states.
		Nondeterministic observations are handled by robot_model_data dict.

		We use sparse matrices to be more memory efficient
		'''
		# O = self.init()

		# preallocate lists for COO construction
		rows = []
		cols = []
		data = []

		# iterate over all valid (x,y) positions on the grid
		for cur_state_id, (x, y, o) in enumerate(self.states):
			# get the observation at the current cell
			obs_at_cur_cell = sensor_model(
											cur_pos=(x,y), 
								  			cur_dir=o, 
											grid=self.grid_obj,
											agent_view_size=self.agent_view_size, 
											see_through_walls=self.see_through_walls)
			
			obs_at_cur_cell_as_tuple = tuple(obs_at_cur_cell.flatten().tolist()) # numpy array are not haahable

			# lookup the observation id
			self.lookup.add(obs_at_cur_cell_as_tuple)
			obs_id = self.lookup[obs_at_cur_cell_as_tuple]

			# set the probability
			# O[cur_state_id, obs_id] = (1 - self.robot_model_data['observations']['corruption_rate'])
			# Append data for COO construction
			rows.append(cur_state_id)
			cols.append(obs_id)
			data.append(1 - self.robot_model_data['observations']['corruption_rate'])

		# Construct the COO matrix
		coo = coo_matrix((data, (rows, cols)), shape=(self.num_states, self.num_observations_upper_bound))

		# convert to CSR sparse matrix format
		O = coo.tocsr()
		return O

class EnvModel:
	'''
	Encapsulate Transition, Obervation and Reward models of an environment.
	NOTE: Reward model is very rudimentary: 
		  - assumes reward cells have a value of 1.0.
		  - complex reward structures are not captured
		  need to implement more sophisticated reward configuration.
	'''
	def __init__(self, 
			  grid: MiniGridEnv, 
			  agent_view_size: int,
			  robot_model_data: dict,
			  valid_actions: set,
			  world_entities:set = {'wall', 'door', 'empty', 'unseen', 'agent'},
			  overlap_entities:set = {'door', 'empty', 'goal', 'agent'},
			  reward_cells: set = {'goal'},
			  see_through_walls:bool = False):
		
		self.robot_model_data = robot_model_data

		# instance of a gridworld with entities in the cells
		self.grid_obj = grid # keep the original Minigrid.core.Grid object
		self.grid = grid.encode()[:,:,0]
		self.see_through_walls = see_through_walls
		self.world_entities = world_entities

		# self.free_cells = np.argwhere(self.grid == OBJECT_TO_IDX['empty'])
		self.overlap_entities = overlap_entities
		self.overlap_entities_int = [OBJECT_TO_IDX[o] for o in self.overlap_entities]
		self.free_cell_logic = np.isin(self.grid, self.overlap_entities_int)
		# get the grid indices in array with shape (M,2) , where M is the number of free cells
		self.free_cells = np.argwhere(self.free_cell_logic)
		self.free_cells_set = {tuple(xy.tolist()) for xy in self.free_cells}
		
		self.num_orientations = len(Directions)
		self.num_states = self.free_cells.shape[0] * self.num_orientations
		
		self.num_entities = len(self.world_entities)
		self.entities_int_list = [OBJECT_TO_IDX[i] for i in world_entities]
	
		self.agent_view_size = agent_view_size
		self.flat_observations_dim = agent_view_size*agent_view_size
		
		self.valid_actions = valid_actions
		self.num_actions = len(valid_actions) 

		self.reward_cells = reward_cells

		# compute based on the number of possible entities in this world 
		# (num of possible entities in cell) ^ {observation dimensions}
		self.num_observations_upper_bound = (self.num_entities)**(self.flat_observations_dim)

		self.transition_model = self.compute_transition_model()
		self.observation_model = self.compute_observation_model()
		self.reward_model = self.generate_reward_lookup()
	
	def compute_transition_model(self) -> TransitionModel:
		# vars() forwards the attributes of self to target object
		return TransitionModel(**vars(self))

	def compute_observation_model(self) -> ObservationModel:
		# vars() forwards the attributes of self to target object
		return ObservationModel(**vars(self), states=self.transition_model.states)
	
	def generate_reward_lookup(self) -> RewardModel:
		return  RewardModel(**vars(self))

	def get_summary(self, include_observation=False):
		print('Environment Model Summary:')
		print('\tnum_free_cells: {:,}'.format(self.free_cells.shape[0]))
		print('\tnum_orientations: {:,}'.format(self.num_orientations))
		print('\tnum_states: {:,}'.format(self.num_states))
		print('\tnum_entities: {:,}'.format(self.num_entities))
		self.transition_model.get_summary()
		self.reward_model.get_summary()
		if include_observation:
			print('\tnum_observations_upper_bound: {:,}'.format(self.num_observations_upper_bound))
			self.observation_model.get_summary()

def sensor_model(cur_pos: Tuple[int, int], 
				 cur_dir: Directions, 
				 grid: Grid, 
				 agent_view_size: int,
				 see_through_walls = False):

	cur_dir_int = cur_dir.value
	grid, vis_mask = gen_obs_grid(cur_pos, cur_dir_int, grid, agent_view_size, see_through_walls)
	observation = grid.encode(vis_mask)
	return observation

def gen_obs_grid(cur_pos: Tuple[int, int], 
				 cur_dir: int, 
				 grid: Grid, 
				 agent_view_size: int,
				 see_through_walls = False):
	'''
	Modified from minigrid.minigrid_env

	Generate the sub-grid observed by the agent.
	This method also outputs a visibility mask telling us which grid
	cells the agent can actually see.
	
	'''
	topX, topY, botX, botY = get_view_exts(cur_pos, cur_dir, agent_view_size)

	grid = grid.slice(topX, topY, agent_view_size, agent_view_size)

	for i in range(cur_dir + 1):
		grid = grid.rotate_left()

	# Process occluders and visibility
	# Note that this incurs some performance cost
	if see_through_walls:
		vis_mask = grid.process_vis(
			agent_pos=(agent_view_size // 2, agent_view_size - 1)
		)
	else:
		vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)
	return grid, vis_mask

def get_view_exts(cur_pos: Tuple[int, int], 
				  cur_dir: int,
				  agent_view_size:int):
	'''
	Modified from minigrid.minigrid_env

	Get the extents of the square set of tiles visible to the agent
	Note: the bottom extent indices are not included in the set
	'''

	agent_view_size = agent_view_size or agent_view_size

	# Facing right
	if cur_dir == 0:
		topX = cur_pos[0]
		topY = cur_pos[1] - agent_view_size // 2
	# Facing down
	elif cur_dir == 1:
		topX = cur_pos[0] - agent_view_size // 2
		topY = cur_pos[1]
	# Facing left
	elif cur_dir == 2:
		topX = cur_pos[0] - agent_view_size + 1
		topY = cur_pos[1] - agent_view_size // 2
	# Facing up
	elif cur_dir == 3:
		topX = cur_pos[0] - agent_view_size // 2
		topY = cur_pos[1] - agent_view_size + 1
	else:
		assert False, "invalid agent direction"

	botX = topX + agent_view_size
	botY = topY + agent_view_size

	return topX, topY, botX, botY

class MonotonicCounter:
	'''
	class to create a dictionary that assigns a monotonically increasing value to each new key.
	we use this to incrementally build the observation matrix.
	that way we avoid constructing the upper bound
	'''

	def __init__(self):
		self._counter = 0
		self._dict = {}

	def add(self, key):
		'''
		Add key to the dictionary with a monotonically increasing value. 
		If the key already exists, does nothing. 
		If the key is new, assigns a monotonically increasing value to it.
		'''
		if key not in self._dict:
			self._dict[key] = self._counter
			self._counter += 1

	def __getitem__(self, key):
		''' Retrieve the counter value for a given key. '''
		return self._dict[key]

	def __contains__(self, key):
		''' Check if a key exists in the dictionary. True if the key exists, False otherwise. '''
		return key in self._dict
	
	def items(self):
		return self._dict.items()
	
	def keys(self):
		return self._dict.keys()