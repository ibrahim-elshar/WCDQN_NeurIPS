# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import bisect

def plots(path, ys, Labels, colors, xLabel, yLabel, Title=None, save=False):
    ''' Plot fuction used to plot results... ys, Labels, colors are lists'''
    plt.figure()
    for i, y in enumerate(ys):
        plt.plot(ys[i], label=Labels[i], color=colors[i], lw=2)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    # plt.legend(loc='best')
    if Title: plt.title(Title)
    # plt.grid()
    if not save:
        plt.show()
    else:
        if Title:
            plt.savefig(path + '/' + Title + '.png')
        else:
            f = input("Please provide a file name: ")
            plt.savefig(path + '/' + f + '.png')


def find_state_indices_binary_search(env,indices):
       # print(len(indices.shape))
       if len(indices.shape)==1:
            # print(tuple(indices),bisect.bisect_left(self.env.stateTable, tuple(indices)))
            return bisect.bisect_left(env.stateTable, tuple(indices))
       else:
            return [bisect.bisect_left(env.stateTable, tuple(i)) for i in indices.tolist()]



def performance_plot(env, Q, ntimes=5):
    env.seed(9)
    returns = 0
    gamma = env.gamma
    episodes_return = []
    for n in range(0, ntimes):
        returns = 0
        discount = 1
        done = False
        state = env.reset()
        state_idx = find_state_indices_binary_search(env,state)

        t = 1
        while not done: #and t <= 100:
            Q[state_idx, :] += (1 - env.action_mask) * -100000000.
            q_values = Q[state_idx, :]
            (state, reward, done, info) = env.step(np.argmax(q_values))
            state_idx = find_state_indices_binary_search(env,state)

            returns += discount * reward
            discount = discount * gamma
            t += 1
        episodes_return.append(returns)
    returns = np.mean(episodes_return)
    std = np.std(episodes_return)
    return returns, std


def plot_performance_curves(env, Q_list, avg_every= 100):
    n = len(Q_list)
    ls =[]
    perf_avg = []
    for i in range(n):
        avg, std = performance_plot(env, Q_list[i])
        ls.append(avg)
        if i!=0 and i % avg_every == 0:
            perf_avg.append(np.mean(ls))
            ls = []
    return perf_avg


def relative_error_plot(num_steps, Qstar, *args):
    Vstar = np.max(Qstar, 1)
    order = None
    Vstar_norm = np.linalg.norm(Vstar, ord=order)
    n = len(args)
    ls = [[] for i in range(n)]
    for i in range(n):
        for j in range(0, num_steps):
            ls[i].append(np.linalg.norm(np.max(args[i][j], 1) - Vstar, ord=order) / Vstar_norm)
    return ls