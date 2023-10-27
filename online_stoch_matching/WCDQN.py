import tensorflow as tf
import numpy as np
import random
import os
import gym
import matplotlib.pyplot as plt
from DQN import DQN, plot_rewards, ReplayBuffer
from collections import deque
from Base import BaseAgent
import logging
import timeit
from numba import jit

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

loss_tracker = tf.keras.metrics.Mean(name="loss")
mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
class CustomModel(tf.keras.Model):
    def train_step(self, data):
        states, q_target, u_layer = data[0]

        with tf.GradientTape() as tape:
            q_pred = self(states, training=True)  # Forward pass
            mse_loss = tf.keras.losses.mse(q_target, q_pred)
            u_loss = tf.keras.backend.mean(tf.keras.backend.square(tf.keras.backend.maximum(q_pred - u_layer, 0.)),
                                           axis=-1)
            loss = mse_loss + 10*u_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        mae_metric.update_state(q_target, q_pred)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result()}

    @property
    def metrics(self):
        return [loss_tracker, mae_metric]


class Network(tf.keras.Model):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_layers_dim,
                 lr=0.0001,
                 loss_mse_tau=1,
                 loss_ub_tau=1,
                 hidden_layers =None,
                 ):
        super().__init__(input_dim, output_dim, hidden_layers_dim, lr)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers_dim = hidden_layers_dim
        self.lr = lr

        self.loss_mse_tau = loss_mse_tau
        self.loss_ub_tau = loss_ub_tau

        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1,
            patience=10, min_lr=0.00000000001)

        if hidden_layers is None:
            hidden_layers = [tf.keras.layers.Dense(dim, activation='relu') for dim in hidden_layers_dim]
        input_layers, q_pred = self.get_layers(hidden_layers = hidden_layers)

        input_state_layer, q_target, u_layer = input_layers

        self.model = tf.keras.models.Model(inputs=input_layers, outputs=q_pred)
        self.model.add_loss(self.bounded_loss_function(q_pred, q_target, u_layer))
        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.lr))

        self.test_model = tf.keras.models.Model(inputs=input_state_layer, outputs=q_pred, name='test_only')

        print('#'*50)
        print('Main Model Summary')
        print('#' * 50)
        print(self.model.summary())
        print('#'*50)
        print('Main Test Model Summary')
        print('#' * 50)
        print(self.test_model.summary())

    def bounded_loss_function(self, q_pred, q_target, u_layer):
        mse_loss = tf.keras.losses.mse(q_target, q_pred)
        u_loss = tf.keras.backend.mean(tf.keras.backend.square(tf.keras.backend.maximum(q_pred - u_layer, 0.)), axis=-1)
        return self.loss_mse_tau * mse_loss + self.loss_ub_tau * u_loss

    def get_layers(self, hidden_layers_dim=None, hidden_layers=None):
        input_state_layer = tf.keras.layers.Input(shape=(self.input_dim,), name='state_in')
        q_target = tf.keras.layers.Input(self.output_dim)
        u_layer = tf.keras.layers.Input(self.output_dim)

        if hidden_layers_dim:
            hidden_layer = input_state_layer
            for i, layer_dim in enumerate(hidden_layers_dim):
                hidden_layer = tf.keras.layers.Dense(layer_dim, activation='relu', name=f'hidden{i}')(hidden_layer)

        elif hidden_layers:
            hidden_layer = input_state_layer
            hidden_layer = tf.keras.layers.Dense(self.hidden_layers_dim[0], activation='relu', name=f'hidden{0}')(hidden_layer)
            for i in range(0, len(hidden_layers)):
                hidden_layer = hidden_layers[i](hidden_layer)


        q_pred = tf.keras.layers.Dense(self.output_dim, activation='linear', name='q_pred')(hidden_layer)

        input_layers = [input_state_layer, q_target, u_layer]

        return input_layers, q_pred

    def predict(self, state, **kwargs):
        return self.test_model(state, **kwargs)
    def fit(self, arr, **kwargs):
        return self.model.fit(arr, **kwargs)
class WCDQN(BaseAgent):
    def __init__(self, env, save_path, model_name,  learning_rate=0.0001, learning_rate_subproblem=0.0001, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                 burn_in=1000, memory_size=10000, batch_size=32, update_subproblem_target_model_freq=32 * 1 * 10, loss_mse_tau = 1, loss_ub_tau = 4,
                 subproblem_hidden_layers_dim=[64, 32, 32],
                 hidden_layers_dim=[64, 32, 32], update_target_model_freq=32 * 1 * 10,
                 experiment_num=1,
                 lambda_max=10, num_lambda=100):
        super().__init__(env, save_path, experiment_num, model_name, learning_rate_subproblem, gamma, epsilon, epsilon_min, epsilon_decay,
                         burn_in, memory_size, batch_size, update_subproblem_target_model_freq,
                         subproblem_hidden_layers_dim, lambda_max, num_lambda)

        self.update_subproblem_target_model_freq = update_subproblem_target_model_freq
        self.subproblem_model, hidden_layers = self._build_subproblem_model(subproblem_hidden_layers_dim)
        self.subproblem_target_model, hidden_layers_target = self._build_subproblem_model(subproblem_hidden_layers_dim)
        self.update_subproblem_target_model()
        self.num_subproblems = env.num_subproblems
        self.loss_mse_tau = loss_mse_tau
        self.loss_ub_tau = loss_ub_tau
        self.hidden_layers_dim = hidden_layers_dim
        self.update_target_model_freq = update_target_model_freq

        self.memory_size = memory_size
        self.memory = Buffer(self.memory_size)

        self.lambda_max =lambda_max
        self.num_lambda = num_lambda
        self.lambda_set = np.linspace(0,self.lambda_max , num=self.num_lambda)

        ##################
        # main network
        print('hidden_layers', hidden_layers)
        self.model = self.target_model = Network(input_dim=self.env.nS,
                                                output_dim=self.env.nA,
                                                hidden_layers_dim=hidden_layers_dim,
                                                lr=learning_rate,
                                                loss_mse_tau=loss_mse_tau,
                                                loss_ub_tau=loss_ub_tau,
                                                hidden_layers=hidden_layers,
                                                )
        print('#' * 50)
        print('subproblem model summary')
        print('#'*50)
        print(self.subproblem_model.summary())
        self.update_target_model()


        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def save_weights(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.subproblem_model.save_weights(os.path.join(self.save_path, 'subproblem_weights.h5'))
        self.model.save_weights(os.path.join(self.save_path, 'weights.h5'))

    def get_actions_upper_bound(self, ):
        subproblem_states = self.env.subproblem_states
        lambdas = self.lambda_set[:, None]
        lambdas_tiled = np.tile(lambdas, (self.env.num_subproblems, 1))
        subproblem_states_repeated = np.repeat(subproblem_states, self.num_lambda, axis=0)
        lambda_subproblem_states = np.concatenate((lambdas_tiled, subproblem_states_repeated), axis=1)
        lambda_subproblem_states_reshaped = lambda_subproblem_states.reshape(-1, lambda_subproblem_states.shape[-1])
        values = self.subproblem_model.predict(lambda_subproblem_states_reshaped, verbose=0)
        values_reshaped = values.reshape(self.env.num_subproblems, self.num_lambda, -1)
        Q = 0
        for i in range(self.env.num_subproblems):
            Q += values_reshaped[i, :, self.env.actions[:, i]] #+ la
        lb = self.env.b/(1-self.gamma) * self.lambda_set
        Q = Q + lb[None, :]
        Q_min = np.min(Q, 1)
        return Q_min

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            u_values = self.get_actions_upper_bound()
            q_values = self.model.predict(np.array([state]))[0]
            return np.argmax(np.minimum(q_values, u_values))


    def evaluate(self):
        state = self.env.reset()
        total_reward = 0
        discount = 1
        done = False
        while not done:
            q_values = self.model.predict(np.array([state]))[0]
            u_values = self.get_actions_upper_bound()
            action = np.argmax(np.minimum(q_values, u_values))
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            total_reward += discount * reward
            discount *= self.gamma
        return total_reward

    def sample_lambdas(self, size):
        lambdas = np.random.choice(self.lambda_set, size=size)
        return lambdas

    def replay(self, morph_cadence = 1):
        if len(self.memory) < self.batch_size:
            return
        initial_time = timeit.default_timer()
        (states, actions, rewards, next_states, dones, subproblem_actions, subproblem_states,
        subproblem_next_states, subproblem_rewards, subproblem_dones) = self.memory.sample(self.batch_size)

        time = timeit.default_timer()
        sample_lambdas = self.sample_lambdas(size=(self.batch_size, self.num_subproblems))

        subproblem_states = np.array(subproblem_states)

        subproblem_states_conc = np.concatenate((sample_lambdas[...,None],subproblem_states), axis=2).reshape(-1, self.env.subproblem_nS+1)

        subproblem_next_states_conc = np.concatenate((sample_lambdas[...,None],subproblem_next_states), axis=2).reshape(-1, self.env.subproblem_nS+1)

        subproblem_actions = np.array(subproblem_actions).ravel()
        subproblem_dones = np.array(subproblem_dones).ravel()
        subproblem_rewards = np.array(subproblem_rewards).ravel()

        subproblem_rewards = subproblem_rewards - (subproblem_actions * sample_lambdas.ravel())

        targets = self.subproblem_model.predict(subproblem_states_conc, verbose=0)
        next_state_targets = self.subproblem_target_model.predict(subproblem_next_states_conc, verbose=0)
        next_state_targets = tf.stop_gradient(next_state_targets)
        max_next_state_targets = np.amax(next_state_targets, axis=1)

        targets[range(self.batch_size*self.num_subproblems), subproblem_actions] = subproblem_rewards + self.gamma * max_next_state_targets * (1 - subproblem_dones)


        history1 = self.subproblem_model.fit(np.array(subproblem_states_conc), np.array(targets), epochs=1, use_multiprocessing=False, verbose=0)

        elapsed = timeit.default_timer() - time


        if self.step % self.update_subproblem_target_model_freq == 0:
            print(f'Step {self.step} : updating subproblem target model...')
            self.update_subproblem_target_model()

        time = timeit.default_timer()
        ub_batch_q_values_arr = self.get_upper_bound(subproblem_states)
        ub_batch_q_values_arr = np.array(ub_batch_q_values_arr, dtype=np.float32)
        elapsed = timeit.default_timer() - time

        time = timeit.default_timer()
        states =np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        actions = np.array(actions)

        main_targets = self.model.predict(states).numpy()

        main_next_state_targets = self.target_model.predict(next_states) 
        main_next_state_targets = tf.stop_gradient(main_next_state_targets)
        max_main_next_state_targets = np.amax(main_next_state_targets, axis=1)

        main_targets[range(self.batch_size), actions] = rewards + self.gamma * max_main_next_state_targets * (1 - dones)

        history = self.model.fit([states, main_targets, ub_batch_q_values_arr], epochs=1, use_multiprocessing=False, verbose=0)
        elapsed = timeit.default_timer() - time



        if self.step % self.update_target_model_freq == 0:
            print(f'Step {self.step} : updating target model...')
            self.update_target_model()
        elapsed = timeit.default_timer() - initial_time
        return history

    def get_upper_bound(self, subproblem_states, num_lambda=6):

        lambda_set = self.sample_lambdas(size=num_lambda)
        lambdas = lambda_set[:, None]
        lambdas_tiled = np.tile(lambdas, (self.batch_size, self.env.num_subproblems, 1))
        subproblem_states_repeated = np.repeat(subproblem_states, num_lambda, axis=1)
        lambda_subproblem_states = np.concatenate((lambdas_tiled, subproblem_states_repeated), axis=2)
        lambda_subproblem_states_reshaped = lambda_subproblem_states.reshape(-1, lambda_subproblem_states.shape[-1])
        values = self.subproblem_model.predict(lambda_subproblem_states_reshaped , verbose=0)
        values_reshaped = values.reshape(self.batch_size, self.env.num_subproblems, num_lambda, -1)
        Q = 0
        for i in range(self.env.num_subproblems):
            Q += values_reshaped[:,i,:,self.env.actions[:,i]] 
        lb = self.env.b / (1 - self.gamma)
        lb=np.array([lb]*subproblem_states.shape[0])[:,None]* lambda_set
        Q = Q + lb[None,:,:]
        Q_min = np.min(Q,2).T

        return Q_min



    def morph_weights(self, smoothing_rate=0.1):
        ''' morphing hidden weights from subproblem model to main model'''
        weights = self.model.get_weights()
        subproblem_weights = self.subproblem_model.get_weights()
        for i in range(1,len(weights)-2):
            weights[i] = weights[i] * (1-smoothing_rate) + subproblem_weights[i] *  smoothing_rate
        self.model.set_weights(weights)


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
                                advertiser_requirement=config['advertiser_requirement'],
                                max_impressions=config['max_impressions'],
                                penalty_vector=config['penalty_vector'],
                                )

    save_path = "online_ad_weights/"


    agent = WCDQN(env, save_path,
                  model_name = 'wcdqn',
                  experiment_num=1,
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
    agent.train(episodes=1000, evaluation_freq=100, evaluation_num_eps=5, save_freq=100, learning_cadence=config['WCDQN']['minibatch_steps_cadence'],
                plot=True, plot_freq=100, training_avg_window=100)

