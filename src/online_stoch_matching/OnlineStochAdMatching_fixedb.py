import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import numba
from numba import jit
import random
import itertools

def sums(length, total_sum):
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in sums(length - 1, total_sum - value):
                yield (value,) + permutation

class OnlineStochAdMatching(gym.Env):
    def __init__(self, seed, episode_limit, num_advertisers, num_impression_types,
                 advertiser_requirement, max_impressions,penalty_vector):

        # Environment name
        self.env_name = 'OnlineStochAdMatching'

        # initialize the environment
        self.seed(seed)  # set the seed for the environment
        self.episode_time = 0  # current period in the episode
        self.num_subproblems = num_advertisers  # num of subproblems

        # parameters for the environment
        self.episode_limit = episode_limit
        self.num_advertisers = num_advertisers
        self.num_impression_types = num_impression_types
        self.noise_length = num_impression_types
        self.advertiser_requirement = np.array(advertiser_requirement)
        self.max_impressions = max_impressions
        self.fixed_num_arriving_impressions = max_impressions
        self.b =  self.fixed_num_arriving_impressions

        self.impressions_space = np.arange(num_impression_types, dtype=np.float32)
        self.impressions_transition_matrix = self.get_impressions_transition_matrix(num_impression_types)

        self.penalty_vector = penalty_vector

        # reward parameters
        np.random.seed(123)
        self.reward_params = np.round(np.random.uniform(0,50, (self.num_advertisers, self.num_impression_types)),2)

        self.nS = self.num_advertisers + 1 + 1 ### 1 because of impression_type and time
        self.subproblem_nS = self.num_advertisers + 2 +1#2 + 1 ### 1 because of impression_type and time

        self.actions = np.array(self.get_actions(self.b))
        self.nA = len(self.actions)
        self.action_index = {tuple(x): index for index, x in enumerate(self.actions)}
        self.actions_indices = np.arange(self.nA)
        self.action_mask = np.ones(self.nA)
        self.action_space = spaces.Discrete(self.nA)

        # subproblem actions
        self.subproblem_actions = np.arange(self.b + 1)
        self.subproblem_nA = len(self.subproblem_actions)

        self.noise_index = numba.typed.Dict()
        self.map_noise_to_max_resource = numba.typed.Dict()
        self.actions_mask = numba.typed.Dict()
        for index, noise in enumerate(self.impressions_space):
            self.noise_index[noise] = index
            self.map_noise_to_max_resource[noise] = self.b
            self.actions_mask[noise] = np.ones(self.nA)

        # observations
        self.low_state = np.zeros(self.num_advertisers)
        self.high_state = self.advertiser_requirement.copy()
        self.low_observation = np.concatenate(([self.impressions_space[0],0], self.low_state), dtype=np.float32)
        self.high_observation = np.concatenate(([self.impressions_space[-1], self.episode_limit], self.high_state), dtype=np.float32)
        self.observation_space = spaces.Box(self.low_observation, self.high_observation, dtype=np.float32)



    def get_impressions_transition_matrix(self, num_impression_types):
        """ Returns the transition matrix for the impressions space. """
        np.random.seed(123)
        impressions_transition_matrix = np.zeros((num_impression_types, num_impression_types))
        low = np.ones(self.num_impression_types)
        high = low * 20
        self.alpha = np.random.uniform(low, high=high, size=(self.num_impression_types, self.num_impression_types))
        for impression in self.impressions_space.astype(int):
            impressions_transition_matrix[impression] = self.np_random.dirichlet(self.alpha[impression])
        return impressions_transition_matrix

    def seed(self, seed=None):
        """ sets the seed for the environment"""
        self.np_random, seed = seeding.np_random(seed)
        self._seed = seed
        self.myRandomPRNG = random.Random(self._seed)
        return [seed]

    def get_observation(self, impression, time, state):
        observation = np.concatenate((np.array([impression, time]), state))
        return observation

    @staticmethod
    def get_subproblem_states(impression, episode_time, state):
        subproblem_states = np.zeros((len(state), len(state) + 3))
        for i in range(len(state)):
            subproblem_states[i] = np.hstack((np.eye(len(state))[i], impression, episode_time, state[i]))
        return subproblem_states

    @staticmethod
    @jit(nopython=True)
    def sample_episode_impressions(initial_impression, seed, episode_limit,
                             impressions_transition_matrix, impressions_space,
                                   ):
        np.random.seed(seed)
        impressions_sample = np.zeros(episode_limit)
        impression = initial_impression
        for i in range(episode_limit):
            impressions_sample[i] = impression
            transition_probability_vector = impressions_transition_matrix[int(impression)]
            impression = impressions_space[np.searchsorted(np.cumsum(transition_probability_vector),
                                                  np.random.random(), side="right")]
        return impressions_sample

    def reset(self):    

        # initialize the state of the episode
        state = self.advertiser_requirement.copy()
        self.initial_impression = self.impressions_space[0]
        impression = self.initial_impression


        # Initialize episode time
        self.episode_time = 0

        # Sample episode impression
        seed = self.np_random.randint(0, 2 ** 32)
        # seed = self.np_random.integers(0, 2 ** 32)
        self.impression_episode_sample = self.sample_episode_impressions(self.initial_impression, seed,
                                                                    self.episode_limit + 1,
                                                                    self.impressions_transition_matrix,
                                                                    self.impressions_space)


        self.observation = self.get_observation(impression,self.episode_time, state)
        self.subproblem_states = np.array(self.get_subproblem_states(impression,self.episode_time, state))
        self.action_mask = np.ones(self.nA) 
        return self.observation

    def get_next_state(self,current_state, action):
        next_state = np.maximum(current_state - action,0)
        return next_state

    def get_actions(self, b):
        """ Returns the actions available in the current state."""
        actions = list(sums(self.num_advertisers, b))
        return actions

    def get_reward(self,impression, state, action):
        """ Returns the reward for the current state and action."""
        r = self.reward_params[:,impression]
        reward = r* np.minimum(state,action)
        return reward


    def step(self, action_index):

        state = self.observation[2:]
        time = self.observation[1]
        noise = int(self.observation[0])
        impression = noise

        subproblem_states = self.subproblem_states.copy()

        action = self.actions[action_index]

        next_state = self.get_next_state(state, action)
        next_state = np.round(next_state, 2)

        self.reward_vector = self.get_reward(impression, state, action)
        self.reward_vector = np.round(self.reward_vector, 2)

        next_noise = self.impression_episode_sample[self.episode_time + 1]

        self.episode_time += 1

        self.observation = self.get_observation(next_noise, self.episode_time, next_state)

        self.subproblem_states = np.array(self.get_subproblem_states(next_noise, self.episode_time, next_state))


        self.action_mask = np.ones(self.nA)

        self.done = bool(self.episode_time == self.episode_limit) or sum(next_state) == 0

        penalty = np.zeros(self.num_advertisers)
        if self.done:
            self.episode_time = 0
            penalty = self.penalty(next_state)

        self.reward_vector = self.reward_vector - penalty

        info = {'noise': noise,
                'next_noise': next_noise,
                'action': action,
                'subproblem_states': subproblem_states,
                'subproblem_next_states': self.subproblem_states,
                'subproblem_rewards': self.reward_vector}


        self.reward = sum(self.reward_vector)

        return self.observation, self.reward, self.done, info

    def penalty(self, state):
        return self.penalty_vector*state

    def sample_action(self):
        actions_probability = self.action_mask / sum(self.action_mask)
        action_index = self.np_random.choice(self.nA, p=actions_probability)
        return action_index


if __name__ == '__main__':

    from online_stoch_matching.config_online_stoch import config

    config = config['OnlineStochAdMatching']


    self = OnlineStochAdMatching(seed=config['seed'],
                                episode_limit=config['episode_limit'],
                                num_advertisers=config['num_advertisers'],
                                num_impression_types=config['num_impression_types'],
                                advertiser_requirement=config['advertiser_requirement'],
                                max_impressions=config['max_impressions'],
                                penalty_vector=config['penalty_vector'],
                                )
    self.reset()
    for _ in range(1):
        self.reset()
        done = False
        while not done:
            action = self.sample_action()
            observation, reward, done, info = self.step(action)
            print('observation:',observation, 'action:',action, 'reward:', reward, 'done:', done, 'info:', info)
            print('#'*100)

    self.close()

