"""
Gym make to stock environment.
"""
import gym
import time
import random
import itertools
import numpy as np
import numba
from numba import jit
import pandas as pd
import scipy.special
from gym import spaces
from gym.utils import seeding
from icecream import ic
from scipy.stats import poisson


def find_tuples(r, k, target):
    """Find all tuples of length k with elements from 0 to r and sum is equal to target."""
    if k == 1:
        return [(i,) for i in range(r + 1) if i == target]
    elif k == 0 or target < 0:
        return []
    else:
        tuples = []
        for i in range(r + 1):
            for t in find_tuples(r, k - 1, target - i):
                tuples.append((i,) + t)
        return tuples


def find_t(r,k,t):
    actions_low = np.zeros(k, int)
    actions_high = np.ones(k, int) * r
    ranges = list(range(x, y) for x, y in zip(actions_low, actions_high + 1))
    actions = np.array(list(itertools.product(*ranges)))
    valid_actions_mask = np.sum(actions, axis=1) <= r
    actions = actions[valid_actions_mask]
    return actions


class MakeToStock(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed, episode_limit, num_products, storage_capacity,
                 holding_cost, backorder_cost, lostsales_cost, max_allowable_backorders,
                 max_resource, lambda_demand, production_rate_coefficients):

        # Environment name
        self.env_name = 'make_to_stock'

        # initialize the environment
        self.seed(seed)  # set the seed for the environment
        self.episode_time = 0  # current period in the episode
        self.current_episode = 0  # current episode
        self.num_subproblems = num_products  # num of subproblems

        # parameters
        self.num_products = num_products  # num of subproblems
        self.storage_capacity = storage_capacity
        self.holding_cost = holding_cost
        self.backorder_cost = backorder_cost
        self.lostsales_cost = lostsales_cost
        self.episode_limit = episode_limit  # maximum number of steps per episode
        self.max_allowable_backorders = np.array(max_allowable_backorders)
        self.max_resource = max_resource
        self.lambda_demand = lambda_demand
        self.production_rate_coefficients = production_rate_coefficients


        self.actions = np.array(find_tuples(self.max_resource, self.num_products, self.max_resource))
        self.action_index = {tuple(x): index for index, x in enumerate(self.actions)}
        self.nA = len(self.actions)


        self.actions_indices = np.arange(self.nA)
        self.action_mask = np.ones(self.nA)
        self.action_space = spaces.Discrete(self.nA)

        # subproblem actions
        self.subproblem_actions = np.arange(self.max_resource + 1)
        self.subproblem_nA = len(self.subproblem_actions)

        # noise
        self.noise_low = 0.8
        self.noise_high = 1.0
        self.noise_delta = 0.05
        self.noise_support = np.round(np.arange(self.noise_low, self.noise_high + self.noise_delta, self.noise_delta),
                                      2)
        self.noise_length = self.noise_support.shape[0]
        # noise transition matrix and action mask
        low = np.ones(self.noise_length)
        high = low * 5
        self.alpha = self.np_random.uniform(low, high=high, size=(self.noise_length, self.noise_length))
        self.noise_transition_matrix = numba.typed.Dict()
        self.noise_index = numba.typed.Dict()
        self.map_noise_to_max_resource = numba.typed.Dict()
        self.actions_mask = numba.typed.Dict()
        for index, noise in enumerate(self.noise_support):
            self.noise_transition_matrix[noise] = self.np_random.dirichlet(self.alpha[index])
            self.noise_index[noise] = index
            self.map_noise_to_max_resource[noise] = self.max_resource
            self.actions_mask[noise] = np.ones(self.nA)

        # observations
        self.low_state = -np.array(max_allowable_backorders)
        self.high_state = np.array(storage_capacity)
        self.low_observation = np.concatenate(([self.noise_low], self.low_state), dtype=np.float32)
        self.high_observation = np.concatenate(([self.noise_high], self.high_state), dtype=np.float32)
        self.observation_space = spaces.Box(self.low_observation, self.high_observation, dtype=np.float32)
        self.nS = self.num_products + 1
        self.subproblem_nS = self.num_products + 2 #2 + 1


        self.normalize_reward = False
        self.min_reward = None
        self.max_reward = None
        self.min_reward_vector = None
        self.max_reward_vector = None


        if self.normalize_reward and (self.min_reward is None or self.max_reward is None):
            self.normalize_reward = False
            self.find_min_max_rewards()
            self.normalize_reward = True

            self.reward_range = self.max_reward - self.min_reward

    def find_min_max_rewards(self, brute_force = False, num_episodes=20000):
        if brute_force:
            min_reward = np.inf
            max_reward = -np.inf
            min_info = None
            max_info = None
            self.normalize_reward = False
            for i in range(num_episodes):
                done = False
                self.reset()
                while not done:
                    action = self.action_space.sample()
                    ns, r, done, info = self.step(action)
                    if r < min_reward:
                        min_reward = r
                        min_info = info
                    if r > max_reward:
                        max_reward = r
                        max_info = info
            self.min_reward = min_reward
            self.max_reward = max_reward
            self.min_reward_vector = min_info['subproblem_rewards']
            self.max_reward_vector = max_info['subproblem_rewards']

        else:
            min_reward, min_reward_vector = self.get_min_reward()
            max_reward, max_reward_vector = 0 , np.zeros(self.num_products)
            self.min_reward = min_reward
            self.max_reward = max_reward
            self.min_reward_vector = min_reward_vector
            self.max_reward_vector = max_reward_vector



    def get_noise(self, noise):
        transition_probability_vector = self.noise_transition_matrix[noise]
        next_noise = self.np_random.choice(self.noise_support, size=None, replace=True, p=transition_probability_vector)
        return np.round(next_noise, 2)

    @staticmethod
    @jit(nopython=True)
    def sample_episode_noise(initial_noise, seed, episode_limit,
                             noise_transition_matrix, noise_support):
        np.random.seed(seed)
        noise_sample = np.zeros(episode_limit)
        noise = initial_noise
        for i in range(episode_limit):
            noise_sample[i] = noise
            transition_probability_vector = noise_transition_matrix[noise]
            noise = noise_support[np.searchsorted(np.cumsum(transition_probability_vector),
                                                  np.random.random(), side="right")]
        return noise_sample

    @staticmethod
    @jit(nopython=True)
    def get_production_rate(action, noise, production_rate_coefficients):
        production_rate = (production_rate_coefficients[0]
                           * action / (action + production_rate_coefficients[1])) * noise
        return production_rate 

    @staticmethod
    @jit(nopython=True)
    def get_next_state(current_state, demand, production_rate, max_allowable_backorders, storage_capacity):
        next_state = current_state - demand + production_rate
        next_state = np.maximum(next_state, -max_allowable_backorders)
        next_state = np.minimum(next_state, storage_capacity)
        return next_state


    @staticmethod
    @jit(nopython=True)
    def get_reward(current_state, demand, production_rate, max_allowable_backorders, storage_capacity,
                   holding_cost_value, backorder_cost_value, lostsales_cost_value, min_reward):
        holding_cost = holding_cost_value * np.maximum(current_state + production_rate, 0)
        backorder_cost = backorder_cost_value * np.maximum(-(current_state + production_rate),0)
        temp = np.maximum(-(current_state + production_rate - demand),0)
        lostsales_cost = lostsales_cost_value * np.maximum(temp - max_allowable_backorders, 0)
        total_cost = holding_cost + backorder_cost + lostsales_cost

        return -total_cost

    def get_min_reward(self):
        self.min_reward = -1
        demand = np.ones(self.num_subproblems) * poisson.isf(0.05, self.lambda_demand)
        production_rate = self.actions[0]
        min_reward = self.get_reward(-self.max_allowable_backorders, demand, production_rate,
                                     self.max_allowable_backorders, self.storage_capacity,
                                     self.holding_cost, self.backorder_cost, self.lostsales_cost, self.min_reward)
        return np.round(sum(min_reward), 2) , min_reward

    @staticmethod
    def get_subproblem_states(noise, state):
        subproblem_states = np.zeros((len(state), len(state) + 2))
        for i in range(len(state)):
            subproblem_states[i] = np.hstack((np.eye(len(state))[i], noise, state[i]))
        return subproblem_states

    def get_observation(self, noise, state):
        observation = np.concatenate((np.array([noise]), state))
        return observation

    def step(self, action_index):
        assert self.action_space.contains(action_index)

        state = self.observation[1:]
        noise = self.observation[0]

        subproblem_states = self.subproblem_states.copy()


        action = self.actions[action_index]
  
        self.demand = self.episode_demand[self.episode_time]

        self.production_rate = self.get_production_rate(action, noise, self.production_rate_coefficients)
  
        next_state = self.get_next_state(state, self.demand, self.production_rate,
                                         self.max_allowable_backorders, self.storage_capacity)
        next_state = np.round(next_state, 2)

        self.reward_vector = self.get_reward(state, self.demand, self.production_rate,
                                             self.max_allowable_backorders, self.storage_capacity,
                                             self.holding_cost, self.backorder_cost,
                                             self.lostsales_cost, self.min_reward)
        self.reward_vector = np.round(self.reward_vector, 2)

        next_noise = self.noise_episode_sample[self.episode_time + 1]  # self.get_noise(noise)


        self.observation = self.get_observation(next_noise, next_state)

        self.subproblem_states = np.array(self.get_subproblem_states(next_noise, next_state))

        self.episode_time += 1

        self.done = bool(self.episode_time == self.episode_limit)

        if self.done:
            self.current_episode += 1
            self.episode_time = 0

        if self.normalize_reward:
            self.reward_vector = (self.reward_vector - self.min_reward_vector)/(self.max_reward_vector - self.min_reward_vector + 1e-8)

        info = {'noise': noise,
                'next_noise': next_noise,
                'action': action,
                'subproblem_states': subproblem_states,
                'subproblem_next_states': self.subproblem_states,
                'subproblem_rewards': self.reward_vector,
                'demand': self.demand,}

        self.reward = sum(self.reward_vector)

        return self.observation, self.reward, self.done, info

    def reset(self):

        # initialize the state of the episode
        state = np.ones(self.num_products) * min(self.storage_capacity)
        self.initial_noise = self.noise_support[0]
        noise = self.initial_noise
        self.observation = self.get_observation(noise, state)
        self.subproblem_states = self.get_subproblem_states(noise, state)

        # Sample episode demand for each product
        self.episode_demand = np.concatenate(
            [self.np_random.poisson(lam=l, size=(self.episode_limit, 1)) for l in self.lambda_demand], 1)

        # Initialize episode time
        self.episode_time = 0

        # Sample episode noise
        seed = self.np_random.randint(0, 2 ** 32)
        # seed = self.np_random.integers(0, 2 ** 32)

        self.noise_episode_sample = self.sample_episode_noise(self.initial_noise, seed,
                                                              self.episode_limit + 1,
                                                              self.noise_transition_matrix,
                                                              self.noise_support)

        return self.observation

    def seed(self, seed=None):
        """ sets the seed for the environment"""
        self.np_random, seed = seeding.np_random(seed)
        self._seed = seed
        self.myRandomPRNG = random.Random(self._seed)
        return [seed]


##########################################################################################

if __name__ == "__main__":

    from Inv_control.conf_1 import config

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

    env.reset()

    for i in range(100):
        done = False
        env.reset()
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print('observation:',observation, 'action:',action, 'reward:', reward, 'done:', done, 'info:', info)
            print('#'*100)

