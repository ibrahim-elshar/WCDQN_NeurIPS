'''
This environment code was borrowed and slightly modified from https://github.com/khalednakhleh/NeurWIN/tree/master/envs
'''


import gym
import time
import random
import itertools 
import numpy as np 
import numba
import pandas as pd 
import scipy.special
from gym import spaces
from gym.utils import seeding

import bisect

from deadlineSchedulingEnv import deadlineSchedulingEnv

class deadlineSchedulingMultipleArmsEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, seed, numEpisodes, batchSize, train, numArms, processingCost, 
        maxDeadline, maxLoad, newJobProb, episodeLimit, scheduleArms, noiseVar, gamma):
        super(deadlineSchedulingMultipleArmsEnv, self).__init__()

        self.seed(seed)
        self.gamma = gamma
        self.time = 0
        self.numEpisodes = numEpisodes
        self.episodeTime = 0 # current period in the episode
        self.currentEpisode = 0
        self.numArms = numArms # num of subproblems
        self.episodeLimit = episodeLimit # total number of  periods in an episode
        self.batchSize = batchSize #specifies after how many episodes the RVs are resampled
        self.state = [] # current state
        self.envs = {} # stores each arm env
        self.noiseVar = noiseVar
        self.newJobProb = newJobProb
        self.train = train #if train == True  update done to reset episode
        self.processingCost = processingCost
        
        self.maxDeadline = maxDeadline
        self.maxLoad = maxLoad
        self.scheduleArms = scheduleArms # how many arms to activate
        self.observationSize = self.numArms*2
        #################### modification for markovian processing cost here
        self.Rmin_single_arm = (-0.2*(self.maxLoad)**2)
        self.Rmin = (-0.2*(self.maxLoad)**2)*self.numArms
        if type(self.processingCost) in [float, int]:
            self.fixedProcessCost = True
            self.Rmax = (1-self.processingCost)*self.numArms
            self.Rmax_single_arm = (1-self.processingCost) 
        else:
            self.fixedProcessCost = False
        if not self.fixedProcessCost:
            self.costToIndex = numba.typed.Dict()
            for i,x in enumerate(self.processingCost):
                self.costToIndex[x] = i
            self.CostTransMatrix = np.array([[0.4, 0.3, 0.3],[0.2, 0.5, 0.3],[0.6, 0.2, 0.2]])
            self.map_bw = {0.2: 2, 0.5:2, 0.8:1} # number of active arms function of cost
            self.nCost = len(self.processingCost)
            self.Rmax = (1-min(self.processingCost))*self.numArms
            self.Rmax_single_arm = (1-min(self.processingCost))
        ####################
        
        self._createActionTable()
        self.subproblem_nA = 2

        maxState = np.tile([self.maxDeadline, self.maxLoad], self.numArms)
        lowState = np.zeros(self.observationSize, dtype=np.float32)
        highState = np.full(self.observationSize, maxState, dtype=np.float32)

        self.state_space = spaces.Box(lowState, highState, dtype=np.float32)

        self.envSeeds = self.G.randint(0, 10000, size=self.numArms)
        self._setTheArms()

        ###############
        self._createStateTable()
        
        self._createSubproblemStateTable()
        ###############
        

    def _createActionTable(self):
        '''function that creates a mapping of actions to take. Will be mapped with the action taken from the agent.'''
        if self.numArms <= 100:
            if self.fixedProcessCost:
                self.actionTable = np.zeros(int(scipy.special.binom(self.numArms, self.scheduleArms)))
                n = int(self.numArms)
                self.actionTable  = list(itertools.product([0, 1], repeat=n))
                self.actionTable = [x for x in self.actionTable if not sum(x) != self.scheduleArms]
                self.nA = len(self.actionTable)
                self.action_space = spaces.Discrete(self.nA )
                self.actions = self.actionTable
                self.actions_idx = list(range(self.nA))
            else:
                self.actionTable = {}
                self.action_space = {}
                self.nAc = {} # size of the action space
                n = int(self.numArms)
                self.actions = []
                prev_scheduleArms = None
                for c in self.processingCost:
                    scheduleArms = self.map_bw[c] 
                    self.actionTable[c]  = list(itertools.product([0, 1], repeat=n))
                    self.actionTable[c] = [x for x in self.actionTable[c] if not sum(x) != scheduleArms]
                    if prev_scheduleArms != scheduleArms:
                        self.actions.extend(self.actionTable[c])
                    prev_scheduleArms = scheduleArms
                    self.nAc[c] = len(self.actionTable[c])
                    self.action_space[c] = spaces.Discrete(self.nAc[c])
                self.nA = len(self.actions)
                
                self.actions_idx_map = {x:i for i,x in enumerate(self.actions)}
                self.actions_mask = numba.typed.Dict()
                for c in self.processingCost:
                    # print(self.nA)
                    self.actions_mask[c] = np.zeros(self.nA)
                    self.actions_mask[c][[self.actions_idx_map[a] for a in self.actionTable[c]]] = 1
        else:
            self.actionTable = None 
            
    def _createStateTable(self):
        ''' function to create a table of all states'''
        self.dimS = self.numArms*2+1
        if self.numArms <= 6:
            if not self.fixedProcessCost:
                A = [self.processingCost]+[ range(x[0],x[1]+1) for x in list(zip(self.state_space.low.astype(int),self.state_space.high.astype(int)))]
            else:
                A = [ range(x[0],x[1]+1) for x in list(zip(self.state_space.low.astype(int),self.state_space.high.astype(int)))]
            self.stateTable = [tuple(x) for x in np.array(list(itertools.product(*A)))] # list(itertools.product(*A))#
            self.nS = len(self.stateTable)
        else:
            self.stateTable = None
            self.nS = None
    
    def _createSubproblemStateTable(self,):
        ''' function to create a table of all states for each subproblem'''
        if not self.fixedProcessCost:
            A = [self.processingCost]+[range(0,self.maxDeadline+1), range(0,self.maxLoad+1)]
        else:
            A = [range(0,self.maxDeadline+1), range(0,self.maxLoad+1)]
        self.subproblemStateTable = [tuple(x) for x in np.array(list(itertools.product(*A)))]
        self.subproblem_dimS = 3 #len(self.subproblemStateTable)
        self.subproblem_nS = len(self.subproblemStateTable)
        
        
    def _calReward(self, action):
        '''Function to calculate recovery function's reward based on supplied state.'''
        if self.actionTable != None:
            if not self.fixedProcessCost:
                c = self.procCost[self.episodeTime]
                self.scheduleArms = self.map_bw[c]
                actionVector = self.actions[action]#self.actionTable[c][action]
            else:
                c = self.processingCost
                actionVector = self.actionTable[action]
        else:
            actionVector = action 
        cumReward = 0
        if not self.fixedProcessCost:
            next_state = [c]
        else:
            next_state = []
        
        envCounter = 0
        #####
        sub_rewards = []
        sub_nextStates = []
        sub_curStates = []
        #####
        for i in range(len(actionVector)):
            ####
            sub_curStates.append((self.state[0],self.envs[envCounter].arm[0][0],self.envs[envCounter].arm[0][1]))
            ####
            if not self.fixedProcessCost:
                self.envs[envCounter].processingCost = c

            if actionVector[i] == 1:
                nextState, reward, done, info = self.envs[envCounter].step(1)
                next_state.append(nextState[0])
                next_state.append(nextState[1])
                cumReward += reward
                #####
                sub_rewards.append(reward)
                sub_nextStates.append((c, nextState[0],nextState[1]))
                #####
            elif actionVector[i] == 0:
                nextState, reward, done, info = self.envs[envCounter].step(0)
                next_state.append(nextState[0])
                next_state.append(nextState[1])
                cumReward += reward
                #####
                sub_rewards.append(reward)
                sub_nextStates.append((c, nextState[0],nextState[1]))
                #####
            envCounter += 1

        nextState = np.array(next_state)
        #####
        sub_rewards = np.array(sub_rewards)
        #####
        return nextState, cumReward, sub_curStates, sub_rewards, sub_nextStates
    
    def isFeasible(self, action):
        return action in np.where(self.action_mask==1)[0]

    def step(self, action):
        ''' Standard Gym function for taking an action. Supplies nextstate, reward, and episode termination signal.'''
        
        assert self.isFeasible(action)
        
        self.time += 1
        
        nextState, reward, sub_curStates, sub_rewards, sub_nextStates = self._calReward(action)
        self.sub_curStates =  sub_nextStates
        self.action_mask = self.actions_mask[nextState[0]]
        self.state = nextState
        
        self.episodeTime += 1
        
        done = bool(self.episodeTime == self.episodeLimit)

        if done: 
            self.currentEpisode += 1
            self.episodeTime = 0

        info = {'action_mask': self.action_mask, 
                'states': sub_curStates,
                'rewards':sub_rewards,
                'nextStates': sub_nextStates}
        # print('#########',info)
        return nextState, reward, done, info

    def _setTheArms(self):
        ''' function that sets the N arms for training'''
        for i in range(self.numArms):
            self.envs[i] = deadlineSchedulingEnv(seed=self.envSeeds[i], numEpisodes=1, episodeLimit=self.episodeLimit, 
                maxDeadline=self.maxDeadline, maxLoad=self.maxLoad, newJobProb=self.newJobProb,
                processingCost=self.processingCost, train=False, batchSize=self.batchSize, noiseVar=self.noiseVar)

    def reset(self):
        ''' Standard Gym function for supplying initial episode state.
        Episodes in the same mini-batch have the same trajectory for valid return comparison.'''

        #### modification for sampling markovian processing cost here 
        if not self.fixedProcessCost:
            currentCost = self.G.choice(self.processingCost)
            self.procCost = [currentCost]
            currentIndex = self.costToIndex[currentCost]
            for i in range(self.episodeLimit-1):
                transProbabilities = self.CostTransMatrix[currentIndex]
                nextCost = self.G.choice(self.processingCost, p=transProbabilities)
                self.procCost.append(nextCost)
                currentIndex = self.costToIndex[nextCost]
            # print(self.procCost)
            self.state = [currentCost]
        else:
            self.state = []
        ##############################       
        

        for i in self.envs:
            vals = self.envs[i].reset()

            val1 = vals[0]
            val2 = vals[1]
            self.state.append(val1)
            self.state.append(val2)
            

        self.state = np.array(self.state)
        self.action_mask = self.actions_mask[self.state[0]]
        
        self.sub_curStates = []
        #####
        for i in self.envs:
            ####
            self.sub_curStates.append((self.state[0],self.envs[i].arm[0][0],self.envs[i].arm[0][1]))
            ####
        return self.state

    def seed(self, seed=None):
        ''' sets the seed for the envirnment'''
        self.np_random, seed = seeding.np_random(seed)
        self._seed = seed
        self.myRandomPRNG = random.Random(self._seed)
        self.G = np.random.RandomState(self._seed) # create a special PRNG for a class instantiation
        return [seed]

    def greedy_policy(self, q_values, action_mask):
        # Creating greedy policy for test time.
        q_values += (1 - action_mask) * -1e15
        action_idx = np.argmax(q_values)
        action = self.actions[action_idx]
        return action, action_idx

    def find_state_indices_binary_search(self, indices):
        return bisect.bisect_left(self.stateTable, tuple(indices))
        if len(indices.shape)==1:
                return bisect.bisect_left(self.stateTable, tuple(indices))
        else:
                return [bisect.bisect_left(self.stateTable, tuple(i)) for i in indices.tolist()]

    def find_action_indices_binary_search(self, indices):
           if len(indices.shape)==1:
                return bisect.bisect_left(self.actions, tuple(indices))
           else:
                return [bisect.bisect_left(self.actions, tuple(i)) for i in indices.tolist()]



    def generate_transition_fn(self, state, actionVector, c):
        '''Function to calculate recovery function's reward based on supplied state.'''
        self.trans = {}
        cumReward = 0
        if not self.fixedProcessCost:
            next_state = [c]
        else:
            next_state = []

        envCounter = 0
        #####
        sub_rewards = []
        sub_nextStates = []
        sub_curStates = []
        #####

        #####
        for i in range(len(actionVector)):
            ####
            sub_curStates.append((state[0], self.envs[envCounter].arm[0][0], self.envs[envCounter].arm[0][1]))
            self.envs[envCounter].arm[0][0] = state[2*i+1]
            self.envs[envCounter].arm[0][1] = state[2*i+2]

            ####
            if not self.fixedProcessCost:
                self.envs[envCounter].processingCost = c

            if actionVector[i] == 1:
                nextState, reward, done, info = self.envs[envCounter].step(1)
                next_state.append(nextState[0])
                next_state.append(nextState[1])
                cumReward += reward
                #####
                sub_rewards.append(reward)
                sub_nextStates.append((c, nextState[0], nextState[1]))
                #####
            elif actionVector[i] == 0:
                nextState, reward, done, info = self.envs[envCounter].step(0)
                next_state.append(nextState[0])
                next_state.append(nextState[1])
                cumReward += reward
                #####
                sub_rewards.append(reward)
                sub_nextStates.append((c, nextState[0], nextState[1]))
                #####
            envCounter += 1

        nextState = np.array(next_state)  # , dtype=np.float32)
        #####
        sub_rewards = np.array(sub_rewards)
        #####
        return nextState, cumReward, sub_curStates, sub_rewards, sub_nextStates

    def Q_iteration(self, tol=0.8, max_iter=100):
        Q = np.zeros((self.nS, self.nA))
        new_Q = np.zeros((self.nS, self.nA))
        print(len(self.stateTable))
        for i in range(max_iter):
            print('Interaton: ', i)
            for st_idx, state in enumerate(self.stateTable):
                state_idx = self.find_state_indices_binary_search(state)
                w= state[0]
                action_mask = self.actions_mask[w]
                wIndex = self.costToIndex[w]
                transProbabilities = self.CostTransMatrix[wIndex]
                for action_idx,action in enumerate(self.actions):
                    if action in self.actionTable[w]:
                        val = 0
                        for next_w in self.processingCost:
                            next_wIndex = self.costToIndex[next_w]
                            probability = transProbabilities[next_wIndex]
                            next_state, reward, sub_curStates, sub_rewards, sub_nextStates = self.generate_transition_fn(state, action, next_w)
                            next_state_idx = self.find_state_indices_binary_search(next_state)
                            val += probability * (reward + self.gamma * np.max(Q[next_state_idx, :]))
                        new_Q[state_idx, action_idx] = val
                    else:
                        Q[state_idx, action_idx] = -1e15
                        new_Q[state_idx, action_idx] = -1e15

            diff = np.abs(Q - new_Q).max()
            print('diff=',diff)
            if diff < tol:
                return Q
            Q = new_Q.copy()
            print("Iteration: ", i)

        return Q







##########################################################################################
'''
For environment validation purposes, the below code checks if the nextstate, reward matches
what is expected given a dummy action.
'''
if __name__ == '__main__':
    SEED = 0

    env = deadlineSchedulingMultipleArmsEnv(seed=SEED, numEpisodes=5000, batchSize=1,
                                            train=True, numArms=3, processingCost=np.array([0.2, 0.5, 0.8]),
                                            maxDeadline=4,
                                            maxLoad=2, newJobProb=0.7, episodeLimit=1000, scheduleArms=1, noiseVar=0.0,
                                            gamma=0.9)

    env.reset()

    observation = env.reset()

    Q = env.Q_iteration()

    np.save('Q_star.npy',Q)
    
 

