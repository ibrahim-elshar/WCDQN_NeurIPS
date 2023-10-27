import logging
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger, handler

def extract_rewards_from_logfile(filename):
    rewards = []
    with open(filename, 'r') as f:
        for line in f:
            if 'Total Rewards:' in line:
                rewards.append(float(line.split(':')[-1].strip()))
    return np.array(rewards)

def average_rewards(rewards_array, n):
    num_episodes = len(rewards_array)
    num_windows = num_episodes // n
    avg_rewards = np.zeros(num_windows)
    for i in range(num_windows):
        start_idx = i * n
        end_idx = start_idx + n
        window_rewards = rewards_array[start_idx:end_idx]
        avg_rewards[i] = np.mean(window_rewards)
    return avg_rewards

def plot_rewards(n=100, log_path='path'):
    rewards = extract_rewards_from_logfile(log_path)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.show()
    avg_rewards = average_rewards(rewards, n)
    plt.plot(avg_rewards)
    plt.xlabel('Episode (x100)')
    plt.ylabel('Average Reward')
    plt.title('Training Rewards')
    plt.show()

def plot_evaluations(evaluation_num_eps, log_path='path'):
    evaluations = extract_rewards_from_logfile(log_path)
    import matplotlib.pyplot as plt
    plt.plot(evaluations)
    plt.xlabel(f'Evaluation (x{evaluation_num_eps})')
    plt.ylabel('Average Reward')
    plt.title('Evaluations')
    plt.show()

@jit(nopython=True)
def polynomial_learning_rate(n, w=0.4):
    """ Implements a polynomial learning rate of the form (1/n**w)
    n: Integer
        The iteration number
    w: float between (0.5, 1]
    Returns 1./n**w as the rate
    """
    assert n > 0, "Make sure the number of times a state action pair has been observed is always greater than 0 before calling polynomial_learning_rate"

    return 1./n**w

