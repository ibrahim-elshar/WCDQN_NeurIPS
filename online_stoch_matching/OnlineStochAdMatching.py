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
                 advertiser_requirement, max_impressions, binomial_probability,):

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
        self.advertiser_requirement = np.array(advertiser_requirement)
        self.binomial_probability = np.array(binomial_probability)
        self.max_impressions = max_impressions

        self.impressions_space = np.arange(num_impression_types)
        self.impressions_transition_matrix = self.get_impressions_transition_matrix(num_impression_types)
        self.nb = max_impressions + 1

        # reward parameters
        self.reward_params = np.round(np.random.uniform(1,4, (self.num_advertisers, self.num_impression_types)),2)

        self.nS = self.num_advertisers + 2 ### 2 because of impression_type and b
        self.subproblem_nS = 2 + 2 ### 2 because of impression_type and b

        # we need an action mask for the subproblems and for the full problem
        self.subproblem_actions_b = {}
        self.subproblem_actions = []
        self.actions_b = {}
        self.actions_index = {}
        self.nA_b = {}
        self.actions_mask = {}
        self.actions = []

        for b in range(1,self.nb):
            self.subproblem_actions_b[b] = np.arange(b)
            self.actions_b[b] = self.get_actions(b)
            self.actions_index[b] =np.arange(len(self.actions_b[b]))
            self.nA_b[b] = len(self.actions_b[b])
            self.actions.extend(self.actions_b[b])


        self.subproblem_actions = np.arange(self.nb)
        self.nA_subproblem = len(self.subproblem_actions)

        self.actions = np.array(self.actions)

        self.nA = len(self.actions)
        self.actions_idx_map = {tuple(a):i for i,a in enumerate(self.actions)}


        self.actions_mask_subproblem = {}
        for b in range(1,self.nb):
            self.actions_mask_subproblem[b] = np.zeros(self.nb)
            self.actions_mask_subproblem[b][:b+1] = 1

        for b in range(1,self.nb):
            self.actions_mask[b] = np.zeros(self.nA)
            self.actions_mask[b][[self.actions_idx_map[a] for a in self.actions_b[b]]] = 1

    def get_impressions_transition_matrix(self, num_impression_types):
        """ Returns the transition matrix for the impressions space. """
        impressions_transition_matrix = np.zeros((num_impression_types, num_impression_types))
        low = np.ones(self.num_impression_types)
        high = low * 20
        self.alpha = self.np_random.uniform(low, high=high, size=(self.num_impression_types, self.num_impression_types))
        for impression in self.impressions_space:
            impressions_transition_matrix[impression] = self.np_random.dirichlet(self.alpha[impression])
        return impressions_transition_matrix

    def seed(self, seed=None):
        """ sets the seed for the environment"""
        self.np_random, seed = seeding.np_random(seed)
        self._seed = seed
        self.myRandomPRNG = random.Random(self._seed)
        return [seed]

    def get_observation(self, impression,b, state):
        observation = np.concatenate((np.array([impression,b]), state))
        return observation

    @staticmethod
    @jit(nopython=True)
    def get_subproblem_states(impression,b, state):
        subproblem_states = [(i + 1, impression,b, s) for i, s in enumerate(state)]
        return subproblem_states

    @staticmethod
    @jit(nopython=True)
    def sample_episode_impressions(initial_impression, seed, episode_limit,
                             impressions_transition_matrix, impressions_space,
                                   max_impressions, binomial_probability):
        np.random.seed(seed)
        impressions_sample = np.zeros(episode_limit)
        num_impressions_sample = np.zeros(episode_limit)
        impression = initial_impression
        for i in range(episode_limit):
            impressions_sample[i] = impression
            n = max_impressions - 1
            p = binomial_probability[impression]
            Y = np.random.binomial(n,p)
            num_impressions_sample[i] = Y + 1
            transition_probability_vector = impressions_transition_matrix[impression]
            impression = impressions_space[np.searchsorted(np.cumsum(transition_probability_vector),
                                                  np.random.random(), side="right")]
        return impressions_sample, num_impressions_sample

    def reset(self):
        # initialize the state of the episode
        state = self.advertiser_requirement.copy()
        self.initial_impression = self.impressions_space[0]
        impression = self.initial_impression


        # Initialize episode time
        self.episode_time = 0

        # Sample episode impression
        # seed = self.np_random.randint(0, 2 ** 32)
        seed = self.np_random.integers(0, 2 ** 32)
        self.impression_episode_sample, self.num_impressions_sample = self.sample_episode_impressions(self.initial_impression, seed,
                                                                    self.episode_limit + 1,
                                                                    self.impressions_transition_matrix,
                                                                    self.impressions_space, self.max_impressions,
                                                                    self.binomial_probability)
        self.b = self.num_impressions_sample[0]

        self.observation = self.get_observation(impression, self.b, state)
        self.subproblem_states = np.array(self.get_subproblem_states(impression,self.b, state))
        self.action_mask = self.actions_mask[self.b]
        self.action_mask_subproblem = self.actions_mask_subproblem[self.b]
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
        noise = int(self.observation[0])
        impression = noise
        b = self.observation[1]

        subproblem_states = self.subproblem_states.copy()

        action = self.actions[action_index]

        next_state = self.get_next_state(state, action)
        next_state = np.round(next_state, 2)

        self.reward_vector = self.get_reward(impression, state, action)
        self.reward_vector = np.round(self.reward_vector, 2)

        next_noise = self.impression_episode_sample[self.episode_time + 1]  
        self.b = self.num_impressions_sample[self.episode_time + 1]  

        self.observation = self.get_observation(next_noise, self.b, next_state)

        self.subproblem_states = np.array(self.get_subproblem_states(next_noise, self.b, next_state))

        self.episode_time += 1
        self.action_mask = self.actions_mask[self.b]
        self.action_mask_subproblem = self.actions_mask_subproblem[self.b]
        self.done = bool(self.episode_time == self.episode_limit) or sum(next_state) == 0

        if self.done:
            self.episode_time = 0

        info = {'noise': noise,
                 'b': b,
                'action': action,
                'subproblem_states': subproblem_states,
                'subproblem_next_states': self.subproblem_states,
                'subproblem_rewards': self.reward_vector}

        self.reward = sum(self.reward_vector)

        return self.observation, self.reward, self.done, info

    def sample_action(self):
        actions_probability = self.action_mask / sum(self.action_mask)
        action_index = self.np_random.choice(self.nA, p=actions_probability)
        return action_index


if __name__ == '__main__':
    env = OnlineStochAdMatching(seed=1, episode_limit=50, num_advertisers=6, num_impression_types=5,
                                advertiser_requirement=[10, 11, 12, 10, 14, 9], max_impressions=3,
                                binomial_probability=[0.1,0.5,0.8,0.4,0.5,0.9],)
    env.reset()
    for _ in range(1):
        env.reset()
        done = False
        while not done:
            action = env.sample_action()
            observation, reward, done, info = env.step(action)
            print(observation, reward, done, info)

    env.close()


