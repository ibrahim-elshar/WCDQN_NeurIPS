import tensorflow as tf
import numpy as np
import random
import os
import gym
import matplotlib.pyplot as plt
from collections import deque
from DQN import DQN, plot_rewards, ReplayBuffer

from utils import setup_logger, plot_rewards, plot_evaluations, extract_rewards_from_logfile, average_rewards, polynomial_learning_rate

class DDQN(DQN):
    def __init__(self, env, save_path, experiment_num, model_name,
                 learning_rate=0.0001, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995, burn_in=1000,
                 memory_size=10000, batch_size=32, update_target_model_freq=32 * 1 * 10, hidden_layers=None,
                 ):
        super().__init__(env, save_path, model_name, experiment_num, learning_rate, gamma, epsilon, epsilon_min,
                 epsilon_decay, burn_in, memory_size, batch_size, update_target_model_freq, hidden_layers)
        if hidden_layers is None:
            hidden_layers = [64, 32, 32]
        self.hidden_layers = hidden_layers
        

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        transition = self.memory.sample(self.batch_size)
        transition = [np.array(x) for x in transition[:5]] + [transition[5]]
        states, actions, rewards, next_states, dones, infos = transition

        targets = self.model.predict(states, verbose=0)

        next_state_targets = self.model.predict(next_states, verbose=0)
        selected_actions = np.argmax(next_state_targets, axis=1)
        ind = tf.stack([range(self.batch_size), selected_actions], axis=1)

        next_state_targets = tf.gather_nd(self.target_model.predict(next_states, verbose=0), ind)

        targets[range(self.batch_size), actions] = rewards + self.gamma * next_state_targets * (1 - dones)

        history = self.model.fit(np.array(states), np.array(targets), epochs=1, use_multiprocessing=False, verbose=0, )
        if self.step % self.update_target_model_freq == 0:
            print(f'Step {self.step} : updating target model...')
            self.update_target_model()
        return history

if __name__ == '__main__':
    for device in tf.config.list_physical_devices():
        print(f'.{device}')

    from Inv_control.conf_1 import config
    from envs.MaketoStock.MakeToStock import MakeToStock

    plot_while_training = True
    config = config['make_to_stock']

    env = MakeToStock(seed=config['seed'],
                      num_products=config['num_products'],
                      storage_capacity=config['storage_capacity'],
                      holding_cost=config['holding_cost'],
                      backorder_cost=config['backorder_cost'],
                      lostsales_cost=config['lostsales_cost'],
                      episode_limit=config['episode_limit'],
                      max_allowable_backorders=config['max_allowable_backorders'],
                      max_resource=config['max_resource'],
                      lambda_demand=config['lambda_demand'],
                      production_rate_coefficients=config['production_rate_coefficients'],
                      )

    save_path = "maketostock_weights/"




    agent = DDQN(env,
                experiment_num=1,
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

    agent.burn_in_memory()


    agent.train(episodes=300, evaluation_freq=100, evaluation_num_eps=5, save_freq=100, learning_cadence=config['DDQN']['minibatch_steps_cadence'],
                plot=True, plot_freq=100, training_avg_window=100)

