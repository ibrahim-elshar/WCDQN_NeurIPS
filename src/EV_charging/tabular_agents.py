import os
import sys
import gym
import time
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import timeit
import bisect
import numba
from numba import jit

### Import Environment
# sys.path.insert(0,'envs/deadlineScheduling')
from deadlineSchedulingMultipleArmsEnv_varCost_bw import deadlineSchedulingMultipleArmsEnv
###
from plots import *
import copy


### helper functions
@jit(nopython=True)
def polynomial_learning_rate(n, w=0.4):
    """ Implements a polynomial learning rate of the form (1/n**w)
    n: Integer
        The iteration number
    w: float between 
    Returns 1./n**w as the rate
    """
    assert n > 0, "Make sure the number of times a state action pair has been observed is always greater than 0 before calling polynomial_learning_rate"

    return 1./n**w
###
def create_python_dict(items):
    return {k: v for k,v in items}

@numba.njit
def create_numba_dict(items):
    return {k: v for k,v in items}

##############################################################################
    
#                            Q-learning

##############################################################################

class Qlearning():
    '''Implementation of the Q-leaning Agent 
    '''
    
    def __init__(self, ENV,
                       SEED,
                       GAMMA=0.9, 
                       eps_greedy_par=0.4,
                       NUM_EPISODES=1000,
                       ):
        self.env = ENV
        self.env.seed(SEED)
        self.env.gamma = GAMMA
        self.gamma = GAMMA
        self.prng = np.random.RandomState(SEED) #Pseudorandom number generator
        print('SEED_ENV:',SEED,'SEED:',SEED)
        self.lr_func = polynomial_learning_rate
        
        L = np.ones((self.env.nS, self.env.nA)) * self.env.Rmin/(1-self.gamma)
        U = np.ones((self.env.nS, self.env.nA)) * self.env.Rmax/(1-self.gamma)
        self.Q =  self.prng.uniform(L,U) #np.zeros((self.env.nS, self.env.nA))     
        self.count = np.zeros([self.env.nS, self.env.nA])
       
        self.eps_greedy_par = eps_greedy_par
        self.num_episodes = NUM_EPISODES
        
        self.Q_list = []
     
                
        self.save_Q_list = True
        self.eval_performance = True

        self.episode_curve = 100
        
        self.substate_indices_map = numba.typed.Dict()
        for i,x in enumerate(self.env.subproblemStateTable):
            self.substate_indices_map[x]= i
            
        self.state_indices_map = numba.typed.Dict()
        for i,x in enumerate(self.env.stateTable):
            self.state_indices_map[x]= i
         
        self.feasible_actions_indices_map = numba.typed.Dict() 
        for w in self.env.processingCost:
            feasible_actions = np.ma.make_mask(self.env.actions_mask[w])
            self.feasible_actions_indices_map[w] = np.arange(self.env.nA)[feasible_actions]
        
    @staticmethod 
    @jit(nopython=True)
    def find_state_indices_map(self, states):
        return [self.state_indices_map[x] for x in states]
    
    @staticmethod 
    @jit(nopython=True)
    def find_substate_indices_map(self, substates):
        return [self.substate_indices_map[x] for x in substates]
    
    
    def find_state_indices_binary_search(self, indices):
           if len(indices.shape)==1:
                return bisect.bisect_left(self.env.stateTable, tuple(indices))
           else:
                return [bisect.bisect_left(self.env.stateTable, tuple(i)) for i in indices.tolist()]


    def epsilon_greedy_policy(self, q_values, state_idx,action_mask, force_epsilon=None):
         '''Creates epsilon greedy probabilities to actions sample from.
            Uses state visit counts.
         '''               
         eps = None
         if force_epsilon:
             eps = force_epsilon
         else:
             # Decay epsilon, save and use
             d = np.sum(self.count[state_idx,:]) if np.sum(self.count[state_idx,:]) else 1 
             eps = 1./d**self.eps_greedy_par
             self.epsilon = eps
         if self.prng.rand() < eps:
              if not self.env.fixedProcessCost:
                  action_idx = self.prng.choice(range(self.env.nA),p=action_mask/sum(action_mask))
                  action = self.env.actions[action_idx]
              else:
                  action_idx = self.prng.choice(self.env.nA,1)[0]
                  action = self.env.actions[action_idx]
         else:
             action_idx = np.argmax(q_values) 
             action = self.env.actions[action_idx]
         return action, action_idx    
            

    def greedy_policy(self, q_values, action_mask):
        # Creating greedy policy for test time.
        q_values += (1 - action_mask) * -1e15
        action_idx = np.argmax(q_values)
        action = self.env.actions[action_idx]
        return action, action_idx


    def train(self, interactive_plot = True, verbose = True):
        ''' Trains the Q-learning agent'''
        start_time = timeit.default_timer()
        self.Q_list.append(np.copy(self.Q))
             
        rewards = []
        self.avg_rewards = []
        # prev_u = 0
        for episode in range(self.num_episodes):
            rewards_episode = 0
            # track the total time steps in this episode
            step = 0
            # initialize state
            state = self.env.reset()
            state_idx = self.find_state_indices_binary_search(state)
            done = False
            discount = 1
            # keep going until get to the goal state
            while not done and step<=300:
                # choose an action based on epsilon-greedy policy
                #  mask unfeasible actions
                self.Q[state_idx, :] += (1-self.env.action_mask)* -100000000.
                q_values = self.Q[state_idx, :] 
                
                action, action_idx = self.epsilon_greedy_policy(q_values, state_idx,self.env.action_mask)
                self.count[state_idx, action_idx] += 1  
                newState, reward, done, info = self.env.step(action_idx)
                rewards_episode += discount * reward
                discount *= self.gamma
                newState_idx =  self.find_state_indices_binary_search(newState)
                self.lr = self.lr_func(self.count[state_idx, action_idx]) 
                
                # Q-Learning update
                self.Q[state_idx, action_idx] +=  self.lr *(reward + self.gamma* np.max(self.Q[newState_idx, :])\
                                                          - self.Q[state_idx, action_idx])
   
                state = newState
                state_idx = newState_idx
                step += 1
            if verbose:   
                print('Episode:',episode, 'reward:',"{:.2f}".format(rewards_episode), 
                      'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:',state_idx)          
             
            rewards.append(rewards_episode)
            if episode % self.episode_curve == 0:
                self.avg_rewards.append(np.mean(rewards))
                rewards = []
            if episode != 0 and episode % 1000 == 0 and interactive_plot:
                plt.plot(self.avg_rewards)
                plt.show()
            
            if self.save_Q_list:
                self.Q_list.append(np.copy(self.Q))   # append Q every episode    
        elapsed_time = timeit.default_timer() - start_time
        print('Time=',elapsed_time)
        if self.eval_performance:
            self.evaluate_performance(self.Q_list)
            self.test_performance(self.Q_list[-1])
        return  self.Q_list, elapsed_time
            
                
    def evaluate_performance(self, Q_list):
        self.env.reset()
        self.performance_rewards = plot_performance_curves(self.env, Q_list, avg_every=self.episode_curve)

    def test_performance(self, Q):
        self.env.reset()
        self.test_reward, self.test_std = performance_plot(self.env, Q, ntimes=100)
        print('test mean reward:', self.test_reward, 'test std:', self.test_std)

##############################################################################
    
#                              Speedy Q-Learning

##############################################################################
        
class SpeedyQLearning(Qlearning):
    """
    Speedy Q-Learning algorithm.
    "Speedy Q-Learning". Ghavamzadeh et. al.. 2011.
    """
    def __init__(self, ENV, SEED,):
        super(SpeedyQLearning, self).__init__(ENV, SEED,)
        self.Q_old = np.copy(self.Q)

        
    def train(self, interactive_plot = True, verbose= True):
        ''' Trains the SQL-learning agent'''
        start_time = timeit.default_timer()
        self.Q_list.append(np.copy(self.Q))

        rewards = []
        self.avg_rewards = []
        for episode in range(self.num_episodes):
            rewards_episode = 0
            # track the total time steps in this episode
            step = 0
            # initialize state
            state = self.env.reset()
            state_idx = self.find_state_indices_binary_search(state)
            discount = 1
            done = False
            # keep going until get to the goal state
            while not done and step<=300:
                old_q = np.copy(self.Q)

                self.Q[state_idx, :] += (1-self.env.action_mask)* -100000000.
                q_values = self.Q[state_idx, :] 
                
                action, action_idx = self.epsilon_greedy_policy(q_values, state_idx,self.env.action_mask)
                self.count[state_idx, action_idx] += 1  
                next_state, reward, done, info = self.env.step(action_idx)
                rewards_episode += discount * reward
                discount *= self.gamma
                next_state_idx =  self.find_state_indices_binary_search(next_state)

                max_q_cur = np.max(self.Q[next_state_idx, :]) if not done else 0.
                max_q_old = np.max(self.Q_old[next_state_idx, :]) if not done else 0.
            
                target_cur = reward + self.gamma * max_q_cur
                target_old = reward + self.gamma * max_q_old
                
                self.lr = self.lr_func(self.count[state_idx, action_idx]) 
                alpha = self.lr
                q_cur = self.Q[state_idx, action_idx]
            
                self.Q[state_idx, action_idx] = q_cur + alpha * (target_old - q_cur) + (
                1. - alpha) * (target_cur - target_old)

                self.Q_old = np.copy(old_q)
            
                state = next_state
                state_idx = next_state_idx
                step += 1

            if verbose:    
                print('Episode:',episode, 'reward:',"{:.2f}".format(rewards_episode), 
                    'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:',state_idx)          
           
            rewards.append(rewards_episode)
            if episode % self.episode_curve == 0:
                self.avg_rewards.append(np.mean(rewards))
                rewards = []
            if episode != 0 and episode % 1000 == 0 and interactive_plot:
                plt.plot(self.avg_rewards)
                plt.show()
            if self.save_Q_list:    
                self.Q_list.append(np.copy(self.Q)) # append Q every episode 
        elapsed_time = timeit.default_timer() - start_time
        print("Time="+str(elapsed_time))
        if self.eval_performance:
            self.evaluate_performance(self.Q_list)
            self.test_performance(self.Q_list[-1])
        return self.Q_list,  elapsed_time
    
##############################################################################
    
#                                Double Q-Learning

##############################################################################    
    
class DoubleQLearning(Qlearning):
    """
    Double Q-Learning algorithm.
    "Double Q-Learning". Hasselt H. V.. 2010.
    """
    def __init__(self, ENV, SEED):
        super(DoubleQLearning, self).__init__(ENV, SEED)
        self.Qprime = np.copy(self.Q)
        self.countprime = np.copy(self.count)

        
    def train(self, interactive_plot=True,verbose=True):
        ''' Trains the Q-learning agent'''
        start_time = timeit.default_timer()
        self.Q_list.append(np.copy(self.Q))
        
        rewards = []
        self.avg_rewards = []
        for episode in range(self.num_episodes):
            rewards_episode = 0
            # track the total time steps in this episode
            step = 0
            # initialize state
            state = self.env.reset()
            state_idx = self.find_state_indices_binary_search(state)
            discount = 1
            done = False
           # keep going until get to the goal state
            while not done and step<=300:
            # choose an action based on epsilon-greedy policy
                self.Q[state_idx, :] += (1-self.env.action_mask)* -100000000.
                self.Qprime[state_idx, :] += (1-self.env.action_mask)* -100000000.
                q_values = (self.Q[state_idx, :] + self.Qprime[state_idx, :] )/2
                action, action_idx = self.epsilon_greedy_policy(q_values, state_idx,self.env.action_mask)
               
                # execute action
                newState, reward, done, info = self.env.step(action_idx)
                rewards_episode += discount * reward
                discount *= self.gamma
                newState_idx =  self.find_state_indices_binary_search(newState)

                # Double Q-Learning update
                rand = np.random.uniform()
                if  rand < .5:
                    self.count[state_idx, action_idx] += 1 
                    self.lr = self.lr_func(self.count[state_idx, action_idx])
                    self.Q[state_idx, action_idx] +=  self.lr *(reward + self.gamma* np.max(self.Qprime[newState_idx, :])\
                                                          - self.Q[state_idx, action_idx])
                else:
                    self.countprime[state_idx, action_idx] += 1 
                    self.lr = self.lr_func(self.countprime[state_idx, action_idx])
                    self.Qprime[state_idx, action_idx] +=  self.lr *(reward + self.gamma* np.max(self.Q[newState_idx, :])\
                                                          - self.Qprime[state_idx, action_idx])
                state = newState
                state_idx = newState_idx
                step += 1
                    
            if verbose:    
               print('Episode:',episode, 'reward:',"{:.2f}".format(rewards_episode), 
                     'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:',state_idx)          
            
            rewards.append(rewards_episode)
            if episode % self.episode_curve == 0:
                self.avg_rewards.append(np.mean(rewards))
                rewards = []
            if episode != 0 and episode % 1000 == 0 and interactive_plot:
                plt.plot(self.avg_rewards)
                plt.show()

            if self.save_Q_list:
                self.Q_list.append((np.copy(self.Q)+np.copy(self.Qprime))/2) # append Q every episode 

        elapsed_time = timeit.default_timer() - start_time
        print("Time="+str(elapsed_time))
        if self.eval_performance:
            self.evaluate_performance(self.Q_list)
            self.test_performance(self.Q_list[-1])
        return self.Q_list,  elapsed_time 


##############################################################################
    
#                               Bias Corrected Q-leaning

##############################################################################    

class Bias_corrected_QL(Qlearning):
    '''Implementation of a Bias Corrected Q-leaning Agent '''
    
    def __init__(self, ENV, SEED, K = 10):
        super(Bias_corrected_QL, self).__init__(ENV, SEED)
        self.K = K
        self.BR = np.zeros((self.env.nS, self.env.nA))
        self.BT = np.zeros((self.env.nS, self.env.nA))
        self.Rvar = np.zeros((self.env.nS, self.env.nA))    
        self.Rmean = np.zeros((self.env.nS, self.env.nA)) 
        self.n_actions = self.env.nA 
        self.count = np.ones((self.env.nS, self.env.nA)) 
        self.T = np.zeros((self.env.nS, self.env.nA, self.K)).astype(int) 
        
    
    def train(self, interactive_plot= True, verbose = True):
         start_time = timeit.default_timer()
         self.Q_list.append(np.copy(self.Q))
  
         rewards = []
         self.avg_rewards = []
         for episode in range(self.num_episodes):
            rewards_episode = 0
            # track the total time steps in this episode
            step = 0
            # initialize state
            state = self.env.reset()
            state_idx = self.find_state_indices_binary_search(state)
            discount = 1
            done = False
            while not done and step<=300:
                # choose an action based on epsilon-greedy policy
                self.Q[state_idx, :] += (1-self.env.action_mask)* -100000000.
                q_values = self.Q[state_idx, :]            
                action, action_idx = self.epsilon_greedy_policy(q_values, state_idx,self.env.action_mask)
                
                
                self.lr = self.lr_func(self.count[state_idx, action_idx])  #1000/(1000+step)
                # execute action
                newState, reward, done, info = self.env.step(action_idx)
                rewards_episode += discount * reward
                discount *= self.gamma
                newState_idx =  self.find_state_indices_binary_search(newState)
                
                self.T[state_idx,action_idx, :-1] = self.T[state_idx,action_idx, 1:]
                self.T[state_idx,action_idx, -1] = int(newState_idx)

                
                prevMean = self.Rmean[state_idx, action_idx]
                prevVar = self.Rvar[state_idx, action_idx]
                prevSigma = np.sqrt(prevVar/self.count[state_idx, action_idx])
        
                self.Rmean[state_idx, action_idx] = prevMean + (reward - prevMean)/self.count[state_idx, action_idx]
                self.Rvar[state_idx, action_idx] = (prevVar + (reward- prevMean)*(reward - self.Rmean[state_idx, action_idx]))/self.count[state_idx, action_idx]
                
                bM= np.sqrt(2*np.log(self.n_actions +7) - np.log(np.log(self.n_actions + 7)) - np.log(4*np.pi))
                self.BR[state_idx, action_idx]=(np.euler_gamma/bM + bM)*prevSigma
                self.BT[state_idx, action_idx]=self.gamma *(np.max(self.Q[newState_idx,:]) - np.mean(np.max(self.Q[self.T[state_idx,action_idx],:],axis=1)))
                if not done: 
                    delta = self.Rmean[state_idx, action_idx]  + self.gamma * np.max(self.Q[newState_idx, :]) - self.Q[state_idx, action_idx]
                else: 
                    delta = self.Rmean[state_idx, action_idx]  - self.Q[state_idx, action_idx]
                self.BR[state_idx, action_idx] = self.BR[state_idx, action_idx] if self.count[state_idx, action_idx] >=2 else 0.0
                self.BT[state_idx, action_idx] = self.BT[state_idx, action_idx] if self.count[state_idx, action_idx] >=self.K else 0.0
               
                self.Q[state_idx, action_idx] +=   self.lr * (delta -self.BR[state_idx, action_idx] - self.BT[state_idx, action_idx])
    
                self.count[state_idx, action_idx] += 1    
                state = newState
                state_idx = newState_idx
                step += 1
            
            if verbose:    
                print('Episode:',episode, 'reward:',"{:.2f}".format(rewards_episode), 
                      'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:',state_idx)          
              
            rewards.append(rewards_episode)
            if episode % self.episode_curve == 0:
                self.avg_rewards.append(np.mean(rewards))
                rewards = []
            if episode != 0 and episode % 1000 == 0 and interactive_plot:
                plt.plot(self.avg_rewards)
                plt.show()
            if self.save_Q_list:
                self.Q_list.append(np.copy(self.Q))
   
         elapsed_time = timeit.default_timer() - start_time
         print("Time="+str(elapsed_time))
         if self.eval_performance:
             self.evaluate_performance(self.Q_list)
             self.test_performance(self.Q_list[-1])
         return self.Q_list,  elapsed_time  

##############################################################################
    
#                           Weakly Coupled Q-learning

##############################################################################
def Biter(env, tol=1e-12,max_iters=1e10):
    ''' B-iteration'''
    B = np.zeros(len(env.processingCost))
    new_B= np.copy(B)
    iters = 0
    while True:
        for w_idx,w in enumerate(env.processingCost):
                val = 0
                for (newState_idx, prob) in enumerate(env.CostTransMatrix[w_idx]):
                     b = env.map_bw[w]
                     val += prob  * (b + env.gamma * B[newState_idx])
                new_B[w_idx] = val
        if np.sum(np.abs(new_B - B)) < tol:
            B = new_B.copy()
            break    
        B= new_B.copy()
        iters += 1 
    return iters, B  


@jit(nopython=True) 
def compute_ub( w_idx, states_idx, actions,num_lambda,Q_lam, B,lambdaSet,numArms):
    u = np.zeros(num_lambda)
    for i,lam in enumerate(lambdaSet):
        u[i] = lam*B[w_idx] + sum([Q_lam[j][i, states_idx[j], actions[j]] for j in range(numArms)])
    idx = np.argmin(u)
    return u[idx], lambdaSet[idx]

@jit(nopython=True) 
def compute_lb(states_idx,actions,Q_lam,env_numArms):
    return max([Q_lam[j][0, states_idx[j], actions[j]] for j in range(env_numArms)])


@jit(nopython=True) 
def updateQLag(exp, numArms, count_subprob,lambdaSet,Q_lam,gamma):
    states_idx,actions,rewards,nextStates_idx = exp
      # Q-Learning update for each subproblem
    for i in range(numArms):
        count_subprob[i, states_idx[i], actions[i]] += 1 
        lam_lr = 0.0005
        for j,lam in enumerate(lambdaSet):
            Q_lam[i][j, states_idx[i], actions[i]] +=  lam_lr *(rewards[i] - lam * actions[i] + gamma* np.max(Q_lam[i][j,nextStates_idx[i], :])\
                                              - Q_lam[i][j, states_idx[i], actions[i]])
    return Q_lam, count_subprob,lam_lr 

@jit(nopython=True) 
def update_all(stateTable,actions, costToIndex, nA, numArms, num_lambda,Q_lam,
               B,lambdaSet, Q, substate_indices_map,state_indices_map,actions_mask,
               feasible_actions_indices_map): 
    for s in stateTable:
        s_idx =  state_indices_map[s]
        ww = s[0]
        ww_idx = costToIndex[ww]
        feasible_actions_idx = feasible_actions_indices_map[ww]
        
        s_sub_curStates = []
        s_sub_curStates_idx = []
        k1, k2 = 1, 2
        for i in range(numArms):
            s_sub_curStates.append((s[0],s[i+k1],s[i+k2]))
            s_sub_curStates_idx.append( substate_indices_map[(s[0],s[i+k1],s[i+k2])])
            k1 += 1
            k2 += 1

        for a_idx in feasible_actions_idx:
            act = actions[a_idx]
            u, lam = compute_ub(ww_idx, np.array(s_sub_curStates_idx), act,num_lambda,Q_lam, B, lambdaSet,numArms)
            Q[s_idx, a_idx] = np.maximum(np.minimum(u,Q[s_idx, a_idx]),-np.inf)
                         
        return Q

class WCQL(Qlearning):
    '''
    Implementation of the WCQL Agent 
    '''
    def __init__(self, ENV, SEED,lagrangian_policy=False):
        super(WCQL, self).__init__(ENV, SEED)
        self.lagrangian_policy = lagrangian_policy
        self.poly_learning_rate_exp = 0.4
        self.num_lambda = 100
        self.lambda_max =  10
        self.lambdaSet = np.linspace(0, self.lambda_max, num=self.num_lambda)
        self.B_lr_func = polynomial_learning_rate
        self.count_B = np.zeros(self.env.nCost)
        self.count_subprob = np.zeros([self.env.numArms,self.env.subproblem_nS, 2]) 

        if not self.env.fixedProcessCost:
            self.B = np.zeros(self.env.nCost)
        self.Q_lam = numba.typed.Dict()

        L_subproblem = np.ones((self.num_lambda, self.env.subproblem_nS, 2))*self.env.Rmin_single_arm/(1-self.gamma)
        U_subproblem = np.ones((self.num_lambda, self.env.subproblem_nS, 2))*self.env.Rmax_single_arm/(1-self.gamma) 
        QQ = self.prng.uniform(L_subproblem,U_subproblem)
        for i in range(self.env.numArms):
            self.Q_lam[i] =  QQ
            
        self.U =  np.ones((self.env.nS, self.env.nA))*self.env.Rmax/(1-self.gamma)                 

        self.B_list  = []
        self.Q_lam_list = []
    def find_indices_subproblem(self, states):
        return [bisect.bisect_left(self.env.subproblemStateTable, tuple(i)) for i in states]  
        
    def updateB(self, w, w_idx, wNext_idx):
        self.count_B[w_idx] += 1
        self.B_lr = self.B_lr_func(self.count_B[w_idx], w=0.4)
        self.B[w_idx] += self.B_lr*(self.env.map_bw[w] + self.gamma * self.B[wNext_idx] - self.B[w_idx])

    def lagrangian_action(self,w_idx, sub_states_idx):
        feasible_actions_idx = np.arange(self.env.nA)[np.ma.make_mask(self.env.action_mask)]
        upper_bound_value = []
        for a_idx in range(self.env.nA):
            if a_idx in feasible_actions_idx:
                act = self.env.actions[a_idx]
                u, lam = compute_ub(w_idx, np.array(sub_states_idx), act, self.num_lambda, self.Q_lam, self.B,
                                    self.lambdaSet, self.env.numArms)
                upper_bound_value.append(u)
            else:
                upper_bound_value.append(-100000000.)
        return upper_bound_value

    def train(self, interactive_plot = True, verbose = True):
        ''' Trains the Q-learning agent'''
        start_time = timeit.default_timer()
        self.Q_list.append(np.copy(self.Q))
        self.B_list.append(np.copy(self.B))
        self.Q_lam_list.append(copy.deepcopy(dict(self.Q_lam)))
        rewards = []
        self.avg_rewards = []
                  
        self.Uplot = []
        self.Q_array = []
        for episode in range(self.num_episodes):
            rewards_episode = 0
            # track the total time steps in this episode
            step = 0
            # initialize state
            state = self.env.reset()
            sub_curStates = self.env.sub_curStates
            sub_states_idx = self.find_indices_subproblem(sub_curStates)
            # print(state)
            state_idx = self.find_state_indices_binary_search(state)
            w = state[0]
            w_idx = self.env.costToIndex[w]
            # print(state,state_idx)
            done = False
            discount = 1
            cond = True
            # keep going until get to the goal state
            while not done and step<300:
                # choose an action based on epsilon-greedy policy
                self.Q[state_idx, :] += (1-self.env.action_mask)* -100000000.
                q_values = self.Q[state_idx, :]   
                ##########################
                if cond and self.lagrangian_policy:#  
                    upper_bound_value = self.lagrangian_action( w_idx, sub_states_idx)
                            
                    action, action_idx = self.epsilon_greedy_policy(upper_bound_value, state_idx,self.env.action_mask)
                else:
                ##########################     
                    action, action_idx = self.epsilon_greedy_policy(q_values, state_idx,self.env.action_mask)
                self.count[state_idx, action_idx] += 1  
                w = state[0]
                newState, reward, done, info = self.env.step(action_idx)
                wNext = newState[0]
                w_idx = self.env.costToIndex[w]
                wNext_idx = self.env.costToIndex[wNext]
                rewards_episode += discount * reward
                discount *= self.gamma
                newState_idx =  self.find_state_indices_binary_search(newState)
                self.lr = self.lr_func(self.count[state_idx, action_idx]) 
                
                # Q-Learning update
                self.Q[state_idx, action_idx] +=  self.lr *(reward + self.gamma* np.max(self.Q[newState_idx, :])\
                                                          - self.Q[state_idx, action_idx])

                    
                if cond:#        
                       
                    sub_states_idx = self.find_indices_subproblem(info['states'])
                    sub_nextStates_idx = self.find_indices_subproblem(info['nextStates'])
                    exp = sub_states_idx, action, info['rewards'], sub_nextStates_idx 
                    self.Q_lam, self.count_subprob, self.lam_lr = updateQLag(exp, self.env.numArms, self.count_subprob,self.lambdaSet,self.Q_lam,self.gamma)
                    self.updateB(w, w_idx, wNext_idx)

                    feasible_actions_idx = np.arange(self.env.nA)[np.ma.make_mask(self.env.action_mask)]
                    
                    for a_idx in feasible_actions_idx:
                            act = self.env.actions[a_idx]
                            u, lam = compute_ub(w_idx, np.array(sub_states_idx), act,self.num_lambda,self.Q_lam, self.B,self.lambdaSet,self.env.numArms)

                            self.Q[state_idx, a_idx] = np.minimum(u,self.Q[state_idx, a_idx])
                     
                state = newState
                state_idx = newState_idx
                sub_states_idx = sub_nextStates_idx
                w_idx = wNext_idx
                step += 1
            if verbose:   
                if cond:
                    print('Episode:',episode, 'reward:',"{:.2f}".format(rewards_episode), 
                      'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:',state_idx, 'W:',w, 'B:', self.B[w_idx], 'Blr:', self.B_lr, 'lam_lr:',self.lam_lr)          
                else:
                    print('Episode:',episode, 'reward:',"{:.2f}".format(rewards_episode), 
                      'epsilon:', "{:.2f}".format(self.epsilon),'action_id:', action_idx, 'state:',state_idx)          
             
            rewards.append(rewards_episode)
            if episode % self.episode_curve == 0:
                self.avg_rewards.append(np.mean(rewards))
                rewards = []
            if episode != 0 and episode % 1000 == 0 and interactive_plot:
                plt.plot(self.avg_rewards)
                plt.show()

                
            if self.save_Q_list:
                self.Q_list.append(np.copy(self.Q))
                self.Q_lam_list.append(copy.deepcopy(dict(self.Q_lam)))
                self.B_list.append(np.copy(self.B))
        elapsed_time = timeit.default_timer() - start_time
        print('Time=',elapsed_time)
        if self.eval_performance:
            self.evaluate_performance(self.Q_list, self.Q_lam_list, self.B_list)
            self.test_performance(self.Q_list[-1],  self.Q_lam_list[-1], self.B_list[-1])
        return self.Q_list, elapsed_time

    def performance_plot(self, Q_lam, B, ntimes = 5):
        self.env.seed(9)
        returns = 0
        gamma = self.env.gamma
        episodes_return = []
        # print(Q_lam, B)
        for n in range(0, ntimes):
            returns = 0
            discount = 1
            done = False
            state = self.env.reset()
            sub_curStates = self.env.sub_curStates
            sub_states_idx = self.find_indices_subproblem(sub_curStates)
            w = state[0]
            w_idx = self.env.costToIndex[w]
            t = 1
            while not done: #and t <= 100:
                feasible_actions_idx = np.arange(self.env.nA)[np.ma.make_mask(self.env.action_mask)]
                upper_bound_value = []
                for a_idx in range(self.env.nA):
                    if a_idx in feasible_actions_idx:
                        act = self.env.actions[a_idx]
                        u, lam = compute_ub(w_idx, np.array(sub_states_idx), act, self.num_lambda, Q_lam, B,
                                            self.lambdaSet, self.env.numArms)

                        upper_bound_value.append(u)
                    else:
                        upper_bound_value.append(-100000000.)


                (state, reward, done, info) = self.env.step(np.argmax(upper_bound_value))
                sub_curStates = self.env.sub_curStates
                sub_states_idx = self.find_indices_subproblem(sub_curStates)
                w = state[0]
                w_idx = self.env.costToIndex[w]
                returns += discount * reward
                discount = discount * gamma
                t += 1
            episodes_return.append(returns)
        returns = np.mean(episodes_return)
        std = np.std(episodes_return)
        return returns, std

    def evaluate_performance(self, Q_list, Q_lam_list, B_list):
        print('evaluating performance...')
        self.env.reset()
        if not self.lagrangian_policy:
            self.performance_rewards = plot_performance_curves(self.env, Q_list,avg_every=self.episode_curve)
        else:
            n = len(Q_lam_list)
            ls = []
            avg_every = self.episode_curve
            perf_avg = []
            for i in range(n):
                avg, std = self.performance_plot(create_numba_dict(tuple(Q_lam_list[i].items())), B_list[i])
                ls.append(avg)
                if i != 0 and i % avg_every == 0:
                    perf_avg.append(np.mean(ls))
                    ls = []

            self.performance_rewards = perf_avg

    def test_performance(self, Q, Q_lam, B):
        self.env.reset()
        if not self.lagrangian_policy:
            self.test_reward, self.test_std = performance_plot(self.env, Q, ntimes=100)
        else:
            self.test_reward, self.test_std = self.performance_plot(create_numba_dict(tuple(Q_lam.items())), B, ntimes = 100)
        print('test mean reward:', self.test_reward, 'test std:', self.test_std)

class LagQlearning_lag_policy(WCQL):
    '''
    Implementation of the Lagrangian Q-leaning Agent 
    '''
    def __init__(self, ENV, SEED,):
        super(LagQlearning_lag_policy, self).__init__(ENV, SEED,lagrangian_policy=True)
    
