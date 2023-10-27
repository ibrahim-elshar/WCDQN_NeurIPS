import tensorflow as tf
import numpy as np
import random
import os
import gym
import matplotlib.pyplot as plt
from DQN import DQN, plot_rewards, ReplayBuffer
from collections import deque
from utils import setup_logger, plot_rewards, plot_evaluations, extract_rewards_from_logfile, average_rewards, polynomial_learning_rate

class Buffer(ReplayBuffer):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, info):
        self.buffer.append((state, action, reward, next_state, done, info['action'], info['subproblem_states'],
                            info['subproblem_next_states'], info['subproblem_rewards'], [done] * len(info['subproblem_rewards'])))

    def sample(self, batch_size):
        (state, action, reward, next_state, done, subproblem_actions, subproblem_states,
         subproblem_next_states, subproblem_rewards, subproblem_dones) = zip(*random.sample(self.buffer, batch_size))
        return (state, action, reward, next_state, done, subproblem_actions, subproblem_states,
                subproblem_next_states, subproblem_rewards, subproblem_dones)

    def __len__(self):
        return len(self.buffer)


class BaseAgent(DQN):
    def __init__(self, env, save_path, experiment_num, model_name,
                 learning_rate=0.0001, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995, burn_in=1000,
                 memory_size=10000, batch_size=32, update_subproblem_target_model_freq=32 * 1 * 10, hidden_layers=None,
                 lambda_max=10, num_lambda=100):
        super().__init__(env, save_path, model_name, experiment_num, learning_rate, gamma, epsilon, epsilon_min,
                 epsilon_decay, burn_in,
                 memory_size, batch_size, update_subproblem_target_model_freq, hidden_layers)
        if hidden_layers is None:
            hidden_layers = [64, 32, 32]
        self.update_subproblem_target_model_freq = update_subproblem_target_model_freq
        self.model = None
        self.target_model = None
        self.subproblem_model, _hidden_layers = self._build_subproblem_model(hidden_layers)
        print('_hidden_layers', _hidden_layers)
        self.subproblem_target_model, _hidden_layers_target = self._build_subproblem_model(hidden_layers)
        self.update_subproblem_target_model()
        self.num_subproblems = env.num_subproblems

        self.memory_size = memory_size
        self.memory = Buffer(self.memory_size)

        self.lambda_max =lambda_max
        self.num_lambda = num_lambda
        self.lambda_set = np.linspace(0,self.lambda_max , num=self.num_lambda)

        self.resource_value = np.zeros(self.env.noise_length)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _build_subproblem_model(self, hidden_layers=[64, 32, 32]):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(hidden_layers[0], input_shape=(self.env.subproblem_nS+1,), activation='relu'))
        _hidden_layers = []
        for i in range(1, len(hidden_layers)):
            _hidden_layers.append(tf.keras.layers.Dense(hidden_layers[i], activation='relu'))
            model.add(_hidden_layers[-1])
        model.add(tf.keras.layers.Dense(self.env.subproblem_nA, activation=None))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(lr=self.learning_rate))
        return model, _hidden_layers


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_subproblem_target_model(self):
        self.subproblem_target_model.set_weights(self.subproblem_model.get_weights())







