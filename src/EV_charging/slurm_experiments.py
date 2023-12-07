import os
import sys
from tabular_agents  import *
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
### Import Environment
# sys.path.insert(0,'envs/deadlineScheduling')
from deadlineSchedulingMultipleArmsEnv_varCost_bw import deadlineSchedulingMultipleArmsEnv
###


experiment_number = 1

dir_path = os.path.dirname(os.path.realpath(__file__))
results_path = f"{dir_path}/results"
if not os.path.isdir(results_path):
    os.makedirs(results_path, exist_ok=True)
    print(f"Created output directory {results_path}")

experiment_results_path = f"{dir_path}/results/exp{experiment_number}"
if not os.path.isdir(experiment_results_path):
    os.makedirs(experiment_results_path, exist_ok=True)
    print(f"Created output directory {experiment_results_path}")
    
agents = [WCQL, LagQlearning_lag_policy, Qlearning, SpeedyQLearning, DoubleQLearning, Bias_corrected_QL]
labels = ['WCQL','QL_lag_policy','QL', 'SQL', 'Double-QL', 'BCQL']

num_agents = len(agents)

num_experiments = 5

SEEDS = np.arange(1,num_experiments+1).astype(int)

experiments = list(
    zip(np.tile(SEEDS, num_agents), np.repeat(agents, num_experiments), np.repeat(labels, num_experiments)))

##############
# Run experiments
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
SEED = experiments[idx][0]
agent_class = experiments[idx][1]
label = experiments[idx][2]
print(f"INFO: Job with array task id {idx} is using SEED={SEED}, agent={label}")
##############

SEEDS_file_name = f"{experiment_results_path}/seeds.npy"
np.save(SEEDS_file_name, SEEDS)

SEED = int(SEED)
exp_num = SEED
Qlist = []
ET_list = []
plot_rewards = []


env = deadlineSchedulingMultipleArmsEnv(seed=SEED, numEpisodes=5000, batchSize=1,
train=True, numArms=3, processingCost=np.array([0.2, 0.5, 0.8]) , maxDeadline=4,
maxLoad=2, newJobProb=0.7, episodeLimit=100, scheduleArms=1, noiseVar=0.0, gamma=0.9)

env.reset()


print(f'Agent:{label}, Experiment: {exp_num}, Seed: {SEED}')

random.seed(SEED)
np.random.seed(SEED)

clss = agent_class
agent = clss(env,
             SEED=SEED)
res = agent.train()

rewards_file_name = f"{experiment_results_path}/{label}_rewards_seed_{SEED}"
np.save(rewards_file_name, agent.avg_rewards)

performance_rewards_file_name = f"{experiment_results_path}/{label}_performance_rewards_seed_{SEED}"
np.save(performance_rewards_file_name, agent.performance_rewards)

Q_file_name = f"{experiment_results_path}/{label}_Q_seed_{SEED}"
np.save(Q_file_name, res[0][-1])

elapsedtime_file_name = f"{experiment_results_path}/{label}_time_seed_{SEED}"
np.save(elapsedtime_file_name, res[1])

test_rewards_file_name = f"{experiment_results_path}/{label}_test_rewards_seed_{SEED}"
np.save(test_rewards_file_name, np.array([agent.test_reward, agent.test_std]))


