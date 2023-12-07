import os
import numpy as np
import sys
import random
from DQN import DQN
from DDQN import DDQN
from OTDQN import OTDQN
from WCDQN import WCDQN
from sys import platform


if platform == "linux" or platform == "linux2":
    # linux crc
    sys.path.append('../')
    from config_online_stoch import config
    from OnlineStochAdMatching_fixedb import OnlineStochAdMatching

from config_online_stoch import config
from OnlineStochAdMatching_fixedb import OnlineStochAdMatching

config = config['OnlineStochAdMatching']
num_experiments = config['num_experiments']
labels = ['DQN', 'DDQN', 'OTDQN', 'WCDQN']
agents = []
for label in labels:
    if label == 'DQN':
        agents.append(DQN)
    elif label == 'DDQN':
        agents.append(DDQN)
    elif label == 'OTDQN':
        agents.append(OTDQN)
    elif label == 'WCDQN':
        agents.append(WCDQN)


num_agents = len(agents)
SEEDS = np.arange(1, num_experiments + 1).astype(int)
experiments = list(
    zip(np.tile(SEEDS, num_agents), np.repeat(agents, num_experiments), np.repeat(labels, num_experiments)))

########################################################################################################################
for i in range(len(experiments)):    
    idx = i#int(os.environ["SLURM_ARRAY_TASK_ID"])
    SEED = experiments[idx][0]
    agent_class = experiments[idx][1]
    label = experiments[idx][2]
    print(f"INFO: Job with array task id {idx} is using SEED={SEED}, agent={label}")

    environment_name = config['environment_name']
    path_seed = f'results/{environment_name}'
    # Check whether the specified path exists or not
    if not os.path.exists(path_seed):
        # Create a new directory because it does not exist
        os.makedirs(path_seed, exist_ok=True)
        print(f"The new directory {path_seed} is created!")
    SEEDS_file_name = f"{path_seed}/seeds.npy"
    np.save(SEEDS_file_name, SEEDS)


    SEED = int(SEED)
    exp_num = SEED
    env = OnlineStochAdMatching(seed=config['seed'],
                                episode_limit=config['episode_limit'],
                                num_advertisers=config['num_advertisers'],
                                num_impression_types=config['num_impression_types'],
                                advertiser_requirement=config['advertiser_requirement'],  # [10, 11, 12, 10, 14, 9],
                                max_impressions=config['max_impressions'],
                                penalty_vector=config['penalty_vector'],
                                )

    env.reset()

    save_path = "online_ad_weights/"


    print(f'Agent:{label}, Experiment: {exp_num}, Seed: {SEED}')
    random.seed(SEED)
    np.random.seed(SEED)
    clss = agent_class

    if label == 'DQN':
        agent = clss(env,
                    experiment_num=exp_num,
                    save_path=save_path,
                    model_name= 'dqn',
                    learning_rate=config['DQN']['learning_rates']['lr'],
                    gamma=config['discount_factor'],
                    epsilon=config['DQN']['exploration']['exploration_initial_epsilon'],
                    epsilon_min=config['DQN']['exploration']['exploration_final_epsilon'],
                    epsilon_decay=config['DQN']['exploration']['epsilon_decay'],
                    burn_in=config['DQN']['replay_buffer']['transitions_burn_in_size'],
                    memory_size=config['DQN']['replay_buffer']['replay_memory_size'],
                    batch_size=config['DQN']['minibatch_size'],
                    update_target_model_freq=config['DQN']['target_network_update_steps_cadence'],
                    hidden_layers=config['DQN']['layers'], )
    elif label == 'DDQN':
        agent = clss(env,
                    experiment_num=exp_num,
                    save_path=save_path,
                    model_name='ddqn',
                    learning_rate=config['DDQN']['learning_rates']['lr'],
                    gamma=config['discount_factor'],
                    epsilon=config['DDQN']['exploration']['exploration_initial_epsilon'],
                    epsilon_min=config['DDQN']['exploration']['exploration_final_epsilon'],
                    epsilon_decay=config['DDQN']['exploration']['epsilon_decay'],
                    burn_in=config['DDQN']['replay_buffer']['transitions_burn_in_size'],
                    memory_size=config['DDQN']['replay_buffer']['replay_memory_size'],
                    batch_size=config['DDQN']['minibatch_size'],
                    update_target_model_freq=config['DDQN']['target_network_update_steps_cadence'],
                    hidden_layers=config['DDQN']['layers'],)
    elif label == 'OTDQN':
        agent = clss(env,
                    experiment_num=exp_num,
                    save_path=save_path,
                    model_name='otdqn',
                    learning_rate=config['OTDQN']['learning_rates']['lr'],
                    gamma=config['discount_factor'],
                    epsilon=config['OTDQN']['exploration']['exploration_initial_epsilon'],
                    epsilon_min=config['OTDQN']['exploration']['exploration_final_epsilon'],
                    epsilon_decay=config['OTDQN']['exploration']['epsilon_decay'],
                    burn_in=config['OTDQN']['replay_buffer']['transitions_burn_in_size'],
                    memory_size=config['OTDQN']['replay_buffer']['replay_memory_size'],
                    batch_size=config['OTDQN']['minibatch_size'],
                    update_target_model_freq=config['OTDQN']['target_network_update_steps_cadence'],
                    hidden_layers=config['OTDQN']['layers'],)
    elif label == 'WCDQN':
        agent = clss(env, save_path,
                    model_name = 'wcdqn',
                    experiment_num=exp_num,
                    learning_rate=config['WCDQN']['learning_rates']['lr'],
                    learning_rate_subproblem=config['WCDQN']['learning_rates']['lr_subproblem'],
                    gamma=config['discount_factor'],
                    epsilon=config['WCDQN']['exploration']['exploration_initial_epsilon'],
                    epsilon_min=config['WCDQN']['exploration']['exploration_final_epsilon'],
                    epsilon_decay=config['WCDQN']['exploration']['epsilon_decay'],
                    burn_in=config['WCDQN']['replay_buffer']['transitions_burn_in_size'],
                    memory_size=config['WCDQN']['replay_buffer']['replay_memory_size'],
                    batch_size=config['WCDQN']['minibatch_size'],
                    update_target_model_freq=config['WCDQN']['target_network_update_steps_cadence'],
                    update_subproblem_target_model_freq=config['WCDQN']['target_network_update_steps_cadence_subproblem'],
                    loss_mse_tau = config['WCDQN']['loss_mse_tau'],
                    loss_ub_tau = config['WCDQN']['loss_ub_tau'],
                    subproblem_hidden_layers_dim=config['WCDQN']['layers_subproblems'],
                    hidden_layers_dim=config['WCDQN']['layers'],
                    lambda_max=config['WCDQN']['lambda_max'],
                    num_lambda=config['WCDQN']['num_lambda'],)




    agent.burn_in_memory()

    agent.train(episodes=config['num_episodes'], evaluation_freq=100, evaluation_num_eps=5, save_freq=500,
                learning_cadence=config['DQN']['minibatch_steps_cadence'],
                plot=False, plot_freq=1000, training_avg_window=100)

