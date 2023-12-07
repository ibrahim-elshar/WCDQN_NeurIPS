import tensorflow as tf
import numpy as np
import random
import os
import gym
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import seaborn as sns
import sys

sys.path.append('../')
from config_online_stoch import config

config = config['OnlineStochAdMatching']

from utils import setup_logger, plot_rewards, plot_evaluations, extract_rewards_from_logfile, average_rewards, polynomial_learning_rate

labels = ['dqn','ddqn', 'otdqn', 'wcdqn']
col_label = ['DQN', 'Double DQN', 'OTDQN', 'WCDQN']

data = {'Episodes': [],
        'Rewards': [],
        'Algorithm': []
        }
for i,label in enumerate(labels):
    print('label:', label)
    for exp_num in range(1, 6):
        print('exp_num:', exp_num)
        log_path = f'online_ad_weights/{label}_training_logger_exp{exp_num}.log'
        rewards = extract_rewards_from_logfile(log_path)
        avg_rewards = average_rewards(rewards, 100)
        n = len(avg_rewards)
        data['Algorithm'].extend([col_label[i]]*n)
        data['Episodes'].extend(np.arange(len(avg_rewards)))
        data['Rewards'].extend(avg_rewards)



print('creating dataframe')
df = pd.DataFrame(data)
print(df)
plt.figure()
sns.set(font_scale=1.5)
sns.set_style('whitegrid')
sns_plot = sns.lineplot(x="Episodes",
                        y="Rewards",
                        hue='Algorithm',
                        data=df,
                        errorbar=('ci', 95),
                        palette=sns.color_palette("tab10", n_colors=len(col_label)),
                        hue_order=col_label,
                        lw=2,
                        )


sns_plot.legend()
plt.xlabel('Episodes (x100)')
plt.ylabel('Total Reward')
fig1 = sns_plot.get_figure()
fig1.tight_layout()
plot_path = 'plots'
if not os.path.isdir(plot_path):
    os.mkdir(plot_path)
    print(f"Created output directory {plot_path}")
fig1.savefig(f'{plot_path}/plots.png')

