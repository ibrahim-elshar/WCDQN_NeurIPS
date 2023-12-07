import tensorflow as tf
import numpy as np
import random
import os
import gym
import matplotlib.pyplot as plt
from collections import deque

from utils import setup_logger, plot_rewards, plot_evaluations, extract_rewards_from_logfile, average_rewards, polynomial_learning_rate


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, info):
        self.buffer.append((state, action, reward, next_state, done, info))

    def sample(self, batch_size):
        state, action, reward, next_state, done, info = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done, info

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(self, env, save_path, model_name, experiment_num=1, learning_rate=0.0001, gamma=0.99, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.9995, burn_in=1000,
                 memory_size=10000, batch_size=32, update_target_model_freq=32 * 1 * 10, hidden_layers=None,
                 ):
        if hidden_layers is None:
            hidden_layers = [64, 32, 32]
        self.env = env
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_target_model_freq = update_target_model_freq
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.hidden_layers = hidden_layers
        self.memory = ReplayBuffer(self.memory_size)
        self.batch_size = batch_size
        self.experiment_num = experiment_num
        self.model = self._build_model(self.hidden_layers)
        self.target_model = self._build_model(self.hidden_layers)
        self.update_target_model()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.model_name = model_name

        self.training_logger_path = save_path + self.model_name + f'_training_logger_exp{self.experiment_num}.log'

        self.training_logger, self.handler = setup_logger('training_logger', self.training_logger_path)


    def _build_model(self, hidden_layers=None):
        if hidden_layers is None:
            hidden_layers = [64, 32, 32]
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(hidden_layers[0], input_shape=(self.env.observation_space.shape[0],),
                                        activation='relu'))
        for i in range(1, len(hidden_layers)):
            model.add(tf.keras.layers.Dense(hidden_layers[i], activation='relu'))
        model.add(tf.keras.layers.Dense(self.env.action_space.n, activation=None))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(np.array([state]), verbose=0)[0])

    def remember(self, state, action, reward, next_state, done, info):
        self.memory.push((state, action, reward, next_state, done, info))

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        print("Burning in memory ...", self.burn_in, "samples to collect.")
        state = self.env.reset()
        for _ in range(self.burn_in):
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            self.memory.push(state, action, reward, next_state, done, info)
            if not done:
                state = next_state
            else:
                state = self.env.reset()
        print("Burn-in complete.")

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        transition = self.memory.sample(self.batch_size)
        transition = [np.array(x) for x in transition[:5]] + [transition[5]]
        states, actions, rewards, next_states, dones, infos = transition

        targets = self.model.predict(states, verbose=0)
        next_state_targets = self.target_model.predict(next_states, verbose=0)
        max_next_state_targets = np.amax(next_state_targets, axis=1)

        targets[range(self.batch_size), actions] = rewards + self.gamma * max_next_state_targets * (1 - dones)

        history = self.model.fit(np.array(states), np.array(targets), epochs=1, use_multiprocessing=False, verbose=0, )
        if self.step % self.update_target_model_freq == 0:
            print(f'Step {self.step} : updating target model...')
            self.update_target_model()
        return history

    def train(self, episodes, evaluation_freq=100, evaluate = True, evaluation_num_eps=5, save_freq=100, learning_cadence=32,
              plot=True, plot_freq=100, training_avg_window=100):
        self.evaluation_num_eps = evaluation_num_eps
        self.evaluation_freq = evaluation_freq
        self.save_freq = save_freq
        self.traning_avg_window = training_avg_window
        self.step = 0
        for episode in range(episodes):
            state = self.env.reset()
            discount = 1
            total_reward = 0
            done = False
            loss = [0]
            history = None
            while not done:
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                self.memory.push(state, action, reward, next_state, done, info)
                state = next_state
                total_reward += reward * discount
                discount *= self.gamma
                if self.step % learning_cadence == 0:
                    history = self.replay()
                if history:
                    loss.append(history.history['loss'][-1])

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                self.step += 1
            self.training_logger.info(
                "Episode {}/{}, Epsilon: {}, Total Rewards: {}".format(episode + 1, episodes, self.epsilon,
                                                                       total_reward))
            print("Episode {}/{}, Epsilon: {}, Total Rewards: {}, Loss: {}".format(episode + 1, episodes, self.epsilon,
                                                                         total_reward, np.mean(loss)))

            if (episode + 1) % save_freq == 0:
                self.save_weights()
            if plot and (episode + 1) % plot_freq == 0:
                plot_rewards(n=training_avg_window, log_path=self.training_logger_path)
        self.handler.close()
        
    def evaluate(self):
        state = self.env.reset()
        discount = 1
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(self.model.predict(np.array([state]), verbose=0)[0])
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            total_reward += reward * discount
            discount *= self.gamma
        return total_reward

    def save_weights(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.model.save_weights(os.path.join(self.save_path, 'weights.h5'))

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)
        self.target_model.load_weights(weights_path)


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


    agent = DQN(env,
                experiment_num=1,
                save_path=save_path,
                model_name='dqn',
                learning_rate=config['DQN']['learning_rates']['lr'],
                gamma=config['discount_factor'],
                epsilon=config['DQN']['exploration']['exploration_initial_epsilon'],
                epsilon_min=config['DQN']['exploration']['exploration_final_epsilon'],
                epsilon_decay=config['DQN']['exploration']['epsilon_decay'],
                burn_in=config['DQN']['replay_buffer']['transitions_burn_in_size'],
                memory_size=config['DQN']['replay_buffer']['replay_memory_size'],
                batch_size=config['DQN']['minibatch_size'],
                update_target_model_freq=config['DQN']['target_network_update_steps_cadence'],
                hidden_layers=config['DQN']['layers'],)

    agent.burn_in_memory()


    agent.train(episodes=5000, evaluate= False, evaluation_freq=100, evaluation_num_eps=5, save_freq=100, learning_cadence=config['DQN']['minibatch_steps_cadence'],
                plot=True, plot_freq=100, training_avg_window=100)
