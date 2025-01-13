
# hide pygame message and gymnasium warnings
import os
import warnings
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium')


from confs.definitions import (WorldModelEnvConfig, 
                            WorldModelTrainingConfig, 
                            WorldModelConfig,
                            DynaQConfig,
                            ModelBasedTrainingConfig)

'''' 
    Instantiate configurations used throughtout tutorial 

'''
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# 00-environment.ipynb

env_config = WorldModelEnvConfig(width=7, height=9, 
                                    agent_view_size=5, 
                                    highlight=True, 
                                    curriculum_learning_prob=0.0, 
                                    compute_env_model=False, 
                                    nondeterministic=False, 
                                    robot_model_data=None,
                                    randomize=False, randomize_agent=False, 
                                    fully_observable=False)

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# 01-mdp.ipynb

robot_model_data = {
    'transitions': {
        'forward': {
            'intended': 0.8,   # Probability of moving {action} as intended
            'slip_left': 0.1, 
            'slip_right': 0.1 
        },
        'left': {
            'intended': 0.8,
            'slip_forward': 0.1,
            'slip_right': 0.1
        },
        'right': {
            'intended': 0.8,
            'slip_forward': 0.1,
            'slip_left': 0.1 
        }
    },
    'observations' : {
        'corruption_rate': 0.0 # how often an observation will be corrupted, must be in range [0,1]
	} 
}

mdp_config = WorldModelEnvConfig(width=7, height=9, 
                                 agent_view_size=5, 
                                 highlight=False, 
                                 curriculum_learning_prob=0.0, 
                                 compute_env_model=True, 
                                 nondeterministic=True, 
                                 robot_model_data=robot_model_data, 
                                 randomize=False, randomize_agent=True, 
                                 fully_observable=True)

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# 03-learning_world_models.ipynb

wm_env_config = WorldModelEnvConfig(width=7, height=9, 
                                 agent_view_size=5, 
                                 highlight=True, 
                                 curriculum_learning_prob=0.6, 
                                 compute_env_model=False, 
                                 randomize=True, randomize_agent=True, 
                                 fully_observable=False)

eval_wm_env_config = WorldModelEnvConfig(width=7, height=9, 
                                 agent_view_size=5, 
                                 highlight=True, 
                                 curriculum_learning_prob=0.0, 
                                 compute_env_model=False, 
                                 randomize=True, randomize_agent=False, 
                                 fully_observable=False)

# use defaults unless specified
wm_training_config = WorldModelTrainingConfig()
wm_config = WorldModelConfig()
dynaq_config = DynaQConfig()
model_based_config = ModelBasedTrainingConfig(dynaq_config,wm_config,wm_training_config)

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------