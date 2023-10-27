# code from Optimality Tightening paper 
# Frank S He, Yang Liu, Alexander G Schwing, and Jian Peng. Learning to play in a day: Faster
# deep reinforcement learning by optimality tightening. arXiv preprint arXiv:1611.01606, 2016.
from opt_tightening_replay_buffer import OptimalityTighteningReplayMemory
import tensorflow as tf
import numpy as np
import random
import os
import gym
import matplotlib.pyplot as plt
from DQN import DQN, plot_rewards, ReplayBuffer
from collections import deque
import logging

from utils import setup_logger, plot_rewards, plot_evaluations, extract_rewards_from_logfile, average_rewards, polynomial_learning_rate



class OTDQN(DQN):
    def __init__(self, env, save_path, experiment_num, model_name,
                 learning_rate=0.0001, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995, burn_in=1000,
                 memory_size=10000, batch_size=32, update_target_model_freq=32 * 1 * 10, hidden_layers=None,
                 ):
        super().__init__(env, save_path, model_name, experiment_num, learning_rate, gamma, epsilon, epsilon_min,
                 epsilon_decay, burn_in, memory_size, batch_size, update_target_model_freq, hidden_layers)
        if hidden_layers is None:
            hidden_layers = [64, 32, 32]
        self.hidden_layers = hidden_layers

        self.transition_len = 4
        self.transition_range = 10

        self.prng = np.random.RandomState(1)  # Pseudorandom number generator

        self.memory = OptimalityTighteningReplayMemory(state_size=self.env.reset().shape[0], rng=self.prng,
                                                       max_steps=memory_size, phi_length=1, discount=gamma,
                                                       batch_size=batch_size,
                                                       transitions_len=self.transition_len)

        self.memory.burn_in = burn_in

        self.batch_size = batch_size

        self.double_dqn = False

        self.close2 = False

        self.start_index = 0
        self.terminal_index = None

        self.step_counter = None

        self.phi_length = 1

        self.penalty_method = "max"

        self.two_train = False
        self.same_update = False

        self.weight_max = 0.8
        self.weight_min = 0.8
        self.weight = self.weight_max
        self.weight_decay_length = 5000000
        self.weight_decay = (self.weight_max - self.weight_min) / self.weight_decay_length

        self.late2 = False
    def burn_in_memory(self):
        print("Burning in memory ...", self.memory.burn_in, "samples to collect.")
        state = self.env.reset()
        for _ in range(self.memory.burn_in):
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            self.memory.add_sample(state, action, reward, done, start_index=self.start_index)
            if not done:
                phi = self.memory.phi(state)
                q_return = np.mean(self.model.predict(phi, verbose=0))
                state = next_state
            else:
                q_return = 0.

                self.start_index = self.memory.top
                self.terminal_index = index = (self.start_index - 1) % self.memory.max_steps

                while True:
                    q_return = q_return * self.gamma + self.memory.rewards[index]
                    self.memory.return_value[index] = q_return
                    self.memory.terminal_index[index] = self.terminal_index
                    index = (index - 1) % self.memory.max_steps
                    if self.memory.terminal[index] or index == self.memory.bottom:
                        break

        print("Burn-in complete.")

    def _do_training(self, step_number):
        if self.close2:
            self.memory.random_close_transitions_batch(self.batch_size, self.transition_len)
        else:
            self.memory.random_transitions_batch(self.batch_size, self.transition_len, self.transition_range)

        target_q_states = np.append(self.memory.forward_states, self.memory.backward_states, axis=1)
        _target_q_states = target_q_states.reshape(target_q_states.shape[0] * target_q_states.shape[1]
                                                   * target_q_states.shape[2], -1)
        target_q_table = self.model.predict(_target_q_states, verbose=0)
        target_q_table = target_q_table.reshape(self.batch_size, target_q_states.shape[1], -1)

        target_double_q_table = None
        if self.double_dqn:
            target_double_q_table = self.model.predict(target_q_states, verbose=0)
        q_values = self.model.predict(self.memory.center_states.reshape(self.batch_size, -1), verbose=0)
        actions = self.memory.center_actions

        ind = tf.stack([np.arange(self.batch_size), actions.ravel()], axis=1)

        q_values = tf.gather_nd(q_values, ind)
        states1 = np.zeros((self.batch_size, self.memory.phi_length, self.memory.state_size), dtype='float32')
        actions1 = np.zeros((self.batch_size, 1), dtype='int32')
        targets1 = np.zeros((self.batch_size, 1), dtype='float32')
        states2 = np.zeros((self.batch_size, self.memory.phi_length, self.memory.state_size), dtype='float32')
        actions2 = np.zeros((self.batch_size, 1), dtype='int32')
        targets2 = np.zeros((self.batch_size, 1), dtype='float32')
        for i in range(self.batch_size):
            q_value = q_values[i]
            if self.two_train:
                states2[i] = self.memory.center_states[i]
                actions2[i] = self.memory.center_actions[i]
                targets2[i] = q_value
            center_position = int(self.memory.center_positions[i])
            if self.memory.terminal.take(center_position, mode='wrap'):
                states1[i] = self.memory.center_states[i]
                actions1[i] = self.memory.center_actions[i]
                targets1[i] = self.memory.center_return_values[i]
                continue
            forward_targets = np.zeros(self.transition_len, dtype=np.float32)
            backward_targets = np.zeros(self.transition_len, dtype=np.float32)
            for j in range(self.transition_len):
                if j > 0 and self.memory.forward_positions[i, j] == center_position + 1:
                    forward_targets[j] = q_value
                else:
                    if not self.double_dqn:
                        forward_targets[j] = self.memory.center_return_values[i] - \
                                             self.memory.forward_return_values[i, j] * self.memory.forward_discounts[
                                                 i, j] + \
                                             self.memory.forward_discounts[i, j] * \
                                             np.max(target_q_table[i, j])
                    else:
                        forward_targets[j] = self.memory.center_return_values[i] - \
                                             self.memory.forward_return_values[i, j] * self.memory.forward_discounts[
                                                 i, j] + \
                                             self.memory.forward_discounts[i, j] * target_double_q_table[i, j]

                if self.memory.backward_positions[i, j] == center_position + 1:
                    backward_targets[j] = q_value
                else:
                    backward_targets[j] = (-self.memory.backward_return_values[i, j] +
                                           self.memory.backward_discounts[i, j] * self.memory.center_return_values[i] +
                                           target_q_table[
                                               i, self.transition_len + j, self.memory.backward_actions[i, j]]) / \
                                          self.memory.backward_discounts[i, j]

            forward_targets = np.append(forward_targets, self.memory.center_return_values[i])
            v0 = v1 = forward_targets[0]
            if self.penalty_method == 'max':
                v_max = np.max(forward_targets[1:])
                v_min = np.min(backward_targets)

                if self.two_train and v_min < q_value:
                    v_min_index = np.argmin(backward_targets)
                    states2[i] = self.memory.backward_states[i, v_min_index]
                    actions2[i] = self.memory.backward_actions[i, v_min_index]
                    targets2[i] = self.memory.backward_return_values[i, v_min_index] - \
                                  self.memory.backward_discounts[i, v_min_index] * self.memory.center_return_values[i] + \
                                  self.memory.backward_discounts[i, v_min_index] * q_value
                if ((self.late2 and self.weight == self.weight_min) or (not self.late2)) \
                        and (v_max - 0.1 > q_value > v_min + 0.1):
                    v1 = v_max * 0.5 + v_min * 0.5
                elif v_max - 0.1 > q_value:
                    v1 = v_max
                elif ((self.late2 and self.weight == self.weight_min) or (not self.late2)) and v_min + 0.1 < q_value:
                    v1 = v_min

            states1[i] = self.memory.center_states[i]
            actions1[i] = self.memory.center_actions[i]
            targets1[i] = v0 * self.weight + (1 - self.weight) * v1

        s1 = states1.reshape(states1.shape[0], -1)
        batch_q_values_arr = self.model.predict(s1, verbose=0)

        indices = tf.stack([np.arange(self.batch_size), actions1.ravel()], axis=1)

        updated_batch_q_values_arr = tf.tensor_scatter_nd_update(batch_q_values_arr, indices, targets1.ravel())

        history = self.model.fit(s1, updated_batch_q_values_arr,
                                 batch_size=self.batch_size, epochs=1, verbose=0)

        if  step_number % self.update_target_model_freq == 0:
            print(f'Step {self.step} : updating target model...')
            self.update_target_model()

        return history

    def train(self, episodes, evaluation_freq=100, evaluation_num_eps=5, save_freq=100, learning_cadence=32,
              plot=True, plot_freq=100, training_avg_window=100):
        self.evaluation_num_eps = evaluation_num_eps
        self.evaluation_freq = evaluation_freq
        self.save_freq = save_freq
        self.traning_avg_window = training_avg_window
        self.step = 0
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                self.memory.add_sample(state, action, reward, done, start_index=self.start_index)
                state = next_state
                total_reward += reward
                if self.step % learning_cadence == 0:
                    history = self._do_training(self.step)
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                self.step += 1
                #
                if not done:
                    phi = self.memory.phi(state)
                    q_return = np.mean(self.model.predict(phi, verbose=0))
                    state = next_state
                else:
                    q_return = 0.

                    self.start_index = self.memory.top
                    self.terminal_index = index = (self.start_index - 1) % self.memory.max_steps

                    while True:
                        q_return = q_return * self.gamma + self.memory.rewards[index]
                        self.memory.return_value[index] = q_return
                        self.memory.terminal_index[index] = self.terminal_index
                        index = (index - 1) % self.memory.max_steps
                        if self.memory.terminal[index] or index == self.memory.bottom:
                            break

            self.training_logger.info(
                "Episode {}/{}, Epsilon: {}, Total Rewards: {}".format(episode + 1, episodes, self.epsilon,
                                                                       total_reward))
            print("Episode {}/{}, Epsilon: {}, Total Rewards: {}".format(episode + 1, episodes, self.epsilon,
                                                                         total_reward))
            if (episode + 1) % save_freq == 0:
                self.save_weights()
            if plot and (episode + 1) % plot_freq == 0:
                plot_rewards(n=training_avg_window, log_path=self.training_logger_path)
        self.handler.close()


if __name__ == '__main__':
    for device in tf.config.list_physical_devices():
        print(f'.{device}')

    from envs.OnlineStochAdMatching.OnlineStochAdMatching_fixedb import OnlineStochAdMatching
    from online_stoch_matching.config_online_stoch import config

    config = config['OnlineStochAdMatching']

    plot_while_training = True

    env = OnlineStochAdMatching(seed=config['seed'],
                                episode_limit=config['episode_limit'],
                                num_advertisers=config['num_advertisers'],
                                num_impression_types=config['num_impression_types'],
                                advertiser_requirement=config['advertiser_requirement'],#[10, 11, 12, 10, 14, 9],
                                max_impressions=config['max_impressions'],
                                penalty_vector=config['penalty_vector'],
                                )

    save_path = "online_ad_weights/"




    agent = OTDQN(env,
                experiment_num=1,
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

    agent.burn_in_memory()


    agent.train(episodes=300, evaluation_freq=100, evaluation_num_eps=5, save_freq=100, learning_cadence=config['OTDQN']['minibatch_steps_cadence'],
                plot=True, plot_freq=100, training_avg_window=100)
