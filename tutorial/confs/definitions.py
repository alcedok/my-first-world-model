
from dataclasses import dataclass, field 
from typing import Set

from minigrid.core.actions import Actions

# Environment Configuration
@dataclass
class WorldModelEnvConfig:
    valid_actions: Set[str] = field(default_factory = lambda: {Actions.left, Actions.right, Actions.forward})
    width: int = 7
    height: int = 9
    highlight: bool = True
    randomize: bool = True
    randomize_agent: bool = True
    max_steps: int = 100
    agent_view_size: int = 5
    see_through_walls: bool = False
    curriculum_learning_prob: float = 0.9 # only set when randomize is on
    compute_env_model: bool = False
    nondeterministic: bool = False
    robot_model_data: dict = None
    fully_observable: bool = False

# Training Configuration
@dataclass
class WorldModelTrainingConfig:
    
    warm_up_rollouts: int = 200
    max_steps: int = field(default = WorldModelEnvConfig().max_steps)

    # warm up 
    epochs: int = 10
    batch_size: int = 10

    initial_learning_rate: float = 1e-2
    grad_clip_norm: float = 100
    learning_rate_gamma: float = 0.9

    temp_anneal: bool = False
    initial_temperature: float = 0.6
    minimum_temperature: float = 0.6
    temperature_anneal_rate: float = 0.05
    
    kl_loss_min_weight: float = 0.01    # NOTE: not implemented  
    kl_loss_init_weight: float = 0.1    # NOTE: not implemented  
    kl_anneal_rate: float = 0.2         # NOTE: not implemented  
    kl_loss_weight: float = 0.1
    
    pred_belief_loss_weight: float = 1.0

    compute_proposed_class_weights: bool = True


# Observation Model Configuration
@dataclass
class ObservationModelConfig:
    categorical_dim: int
    num_categorical_distributions: int
    gumbel_temperature: float
    concept_dim: int = 11
    concept_embed_dim: int = 3
    num_att_heads: int = 2
    conv1_hidden_dim: int = 4

# Transition Model Configuration
@dataclass
class TransitionModelConfig:
    categorical_dim: int
    num_categorical_distributions: int
    gumbel_temperature: float
    action_embed_dim: int
    num_actions: int = field(default_factory = lambda: len(WorldModelEnvConfig().valid_actions))
    fc1_hidden_dim: int = 100

# Reward Model Configuration
@dataclass
class RewardModelConfig:
    categorical_dim: int
    num_categorical_distributions: int
    action_embed_dim: int
    with_action: bool = False
    fc1_hidden_dim: int = 100
    reward_output_dim: int = 1

# World Model Configuration
@dataclass
class WorldModelConfig:
    categorical_dim: int = 6
    num_categorical_distributions: int = 6
    gumbel_hard: bool = False
    gumbel_temperature: float = field(default = WorldModelTrainingConfig().initial_temperature)
    num_actions: int = field(default_factory = lambda: len(WorldModelEnvConfig().valid_actions))
    action_embed_dim: int = 3

    observation_model_config: ObservationModelConfig = field(default = ObservationModelConfig(categorical_dim, num_categorical_distributions, gumbel_temperature))
    transition_model_config: TransitionModelConfig = field(default = TransitionModelConfig(categorical_dim, num_categorical_distributions, gumbel_temperature, action_embed_dim))
    reward_model_config: RewardModelConfig = field(default=RewardModelConfig(categorical_dim, num_categorical_distributions, action_embed_dim))

@dataclass
class DynaQConfig:    
    valid_actions: Set[str] = field(default_factory = lambda: {i for i in WorldModelEnvConfig().valid_actions})
    num_actions: int = field(default_factory = lambda: len(WorldModelEnvConfig().valid_actions))
    qnet_input_dim: int = field(default=(WorldModelConfig().categorical_dim * WorldModelConfig().num_categorical_distributions + WorldModelConfig().action_embed_dim ))
    qnet_fc1_hidden_dim: int = 200
    
    gamma: float = 0.99
    optimizer_learning_rate: float = 0.001
    epsilon: float = 0.1
    num_simulations: int = 10

@dataclass
class ModelBasedTrainingConfig:
    dynaq: DynaQConfig
    world_model: WorldModelConfig
    world_model_training: WorldModelTrainingConfig