import os,sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
### Import Environment
# sys.path.insert(0,'envs/deadlineScheduling')
from deadlineSchedulingMultipleArmsEnv_varCost_bw import deadlineSchedulingMultipleArmsEnv
###

Qstar= np.load('Q_star.npy')
Vstar = np.max(Qstar,1)

order = None
Vstar_norm = np.linalg.norm(Vstar[Vstar!=-1e15],ord=order)

Qstar_norm = np.linalg.norm(Qstar[Qstar !=-1e15])

labels = ['QL', 'Double-QL','SQL', 'BCQL','QL_lag_policy','WCQL']
legend_labels = ['QL', 'Double-QL','SQL', 'BCQL','Lagrangian QL','WCQL']

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


SEEDS_file_name = f"{experiment_results_path}/seeds.npy"

SEEDS = np.load(SEEDS_file_name)

#######
# Training curves
#######
palette_tab10 = sns.color_palette("tab10", 10)
palette = sns.color_palette([palette_tab10[0], palette_tab10[1], palette_tab10[5], palette_tab10[4], palette_tab10[2], palette_tab10[3]])

data = {'Episodes': [],
        'Rewards': [],
        'Algorithm': []
        }
col_labels = []
for label in labels:
    col_label = 'Lagrangian QL' if label == 'QL_lag_policy' else label
    col_labels.append(col_label)
    for SEED in SEEDS:
        rewards_file_name = f"{experiment_results_path}/{label}_rewards_seed_{SEED}.npy"
        experiment_rewards = np.load(rewards_file_name)
        n = len(experiment_rewards)
        data['Algorithm'].extend([col_label] * n)
        data['Episodes'].extend(np.arange(len(experiment_rewards)))
        data['Rewards'].extend(experiment_rewards)

df = pd.DataFrame(data)

legend_labels = col_labels

fig1=plt.figure()
sns.set(font_scale=1)
sns.set_style('whitegrid')
sns_plot = sns.lineplot(x="Episodes",
                        y="Rewards",
                        hue='Algorithm',
                        data=df,
                        ci=95,
                        palette=palette,
                        hue_order=legend_labels,
                        lw=2,
                        )

sns_plot.legend()  
plt.xlabel('Episodes (x100)')
plt.ylabel('Total Reward')
fig1 = sns_plot.get_figure()
fig1.tight_layout()
plot_path = f"{experiment_results_path}/plots"
if not os.path.isdir(plot_path):
    os.makedirs(plot_path, exist_ok=True)
    print(f"Created output directory {plot_path}")
fig1.savefig(f'{plot_path}/exp{experiment_number}_training_plots.png')

df.to_pickle(f'{experiment_results_path}/exp{experiment_number}_training_df.pkl')




#######
# Performance curves
#######
fig2=plt.figure()
data = {'Episodes': [],
        'Rewards': [],
        'Algorithm': [],
        }

data_rel_error = {'Episodes': [],
        'Algorithm': [],
        'Relative Error': [],
        }
col_labels = []
for label in labels:
    col_label = 'Lagrangian QL' if label == 'QL_lag_policy' else label
    col_labels.append(col_label)
    for SEED in SEEDS:
        rewards_file_name = f"{experiment_results_path}/{label}_performance_rewards_seed_{SEED}.npy"
        experiment_rewards = np.load(rewards_file_name)
        n = len(experiment_rewards)
        data['Algorithm'].extend([col_label] * n)
        data['Episodes'].extend(np.arange(len(experiment_rewards)))
        data['Rewards'].extend(experiment_rewards)



df = pd.DataFrame(data)

legend_labels = col_labels

sns.set(font_scale=1)
sns.set_style('whitegrid')
sns_plot = sns.lineplot(x="Episodes",
                        y="Rewards",
                        hue='Algorithm',
                        data=df,
                        ci=95,
                        palette=palette,  
                        hue_order=legend_labels,
                        lw=2,
                        )

sns_plot.legend()  
plt.xlabel('Episodes (x100)')
plt.ylabel('Total Reward')
fig2 = sns_plot.get_figure()
fig2.tight_layout()
plot_path = f"{experiment_results_path}/plots"
if not os.path.isdir(plot_path):
    os.makedirs(plot_path, exist_ok=True)
    print(f"Created output directory {plot_path}")
fig2.savefig(f'{plot_path}/exp{experiment_number}_performance_plots.png')

df.to_pickle(f'{experiment_results_path}/exp{experiment_number}_performance_df.pkl')



fig3=plt.figure()
data_rel_error = {'Episodes': [],
        'Algorithm': [],
        'Relative Error': [],
        }

labels = ['QL', 'Double-QL','SQL', 'BCQL','WCQL']
legend_labels = ['QL', 'Double-QL','SQL', 'BCQL','WCQL']

col_labels = []
for label in labels:
    col_labels.append(col_label)
    for SEED in SEEDS:
        rewards_file_name = f"{experiment_results_path}/{label}_performance_rewards_seed_{SEED}.npy"
        experiment_rewards = np.load(rewards_file_name)
        Q = np.load(f"{experiment_results_path}/{label}_Q_list_seed_{SEED}.npy")
        n = len(Q)
        data_rel_error['Algorithm'].extend([col_label] * n)
        data_rel_error['Episodes'].extend(np.arange(n))
        ls = []
        for i in range(len(Q)):
            ls.append(np.linalg.norm(np.max(Q[i],1)[Vstar!=-1e15]-Vstar[Vstar!=-1e15],ord=order)/Vstar_norm)

        print(ls)
        data_rel_error['Relative Error'].extend(ls)

df_rel_error = pd.DataFrame(data_rel_error)



palette_tab10 = sns.color_palette("tab10", 10)
palette = sns.color_palette([palette_tab10[0], palette_tab10[1], palette_tab10[5], palette_tab10[4], palette_tab10[3]])

sns.set(font_scale=1.3)
sns.set_style('whitegrid')
sns_plot = sns.lineplot(x="Episodes",
                        y="Relative Error",
                        hue='Algorithm',
                        data=df_rel_error,
                        ci=95,
                        palette=palette,  
                        hue_order=legend_labels,
                        lw=2,
                        )

sns_plot.legend()  
plt.xlabel('Episodes (x200)')
plt.ylabel('Relative Error')
fig3 = sns_plot.get_figure()
fig3.tight_layout()
plot_path = f"{experiment_results_path}/plots"
if not os.path.isdir(plot_path):
    os.makedirs(plot_path, exist_ok=True)
    print(f"Created output directory {plot_path}")
fig3.savefig(f'{plot_path}/exp{experiment_number}_re_plots_fontsize_1_3.png')

