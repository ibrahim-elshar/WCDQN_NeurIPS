import numpy as np

# online stochastic matching
###############
num_advertisers = 6
max_impressions = 2

config = {'OnlineStochAdMatching': {'num_experiments': 5,
                            'Labels': ['DQN','DDQN', 'OTDQN', 'WCDQN'],
                            'environment_name': 'OnlineStochAdMatching',
                            'seed': 10,
                            'num_episodes': 10000,
                            'episode_limit': 25,
                            'curve_episode_cadence': 100,
                            'discount_factor': 0.99,
                            ###
                            'num_advertisers': num_advertisers,
                            'num_impression_types': 3,
                            'max_impressions': max_impressions,
                            'advertiser_requirement': np.array([10, 11, 12, 10, 14, 9]), #np.array([10,2,13,4,1,11]),
                            ###
                            'state_dim': num_advertisers + 1,
                            'subproblem_state_dim': 3,
                            'subproblem_num_actions': np.arange(max_impressions + 1),
                            'penalty_vector': np.random.uniform(1,10,size=6),
                            ###
                             'DQN' : {'layers': [ 64,32,],
                                      'learning_rates': {'lr': 0.0001,
                                                         },
                                      'replay_buffer': {
                                                        'transitions_burn_in_size' : 10000,
                                                        'replay_memory_size': 1000000,
                                                        },
                                      'exploration': {'exploration_initial_epsilon' : 1.,
                                                      'exploration_final_epsilon' : 0.05,
                                                      'epsilon_decay': 0.99997,
                                                      },

                                      'minibatch_size':32, 
                                      'target_network_update_steps_cadence' : 32,
                                      'minibatch_steps_cadence' : 4,
                                      },
                            'DDQN': {'layers': [64,32,],
                                     'learning_rates': {'lr': 0.0001,
                                                        },
                                     'replay_buffer': {
                                         'transitions_burn_in_size': 10000,
                                         'replay_memory_size': 1000000,
                                     },
                                     'exploration': {'exploration_initial_epsilon': 1.,
                                                     'exploration_final_epsilon': 0.05,
                                                     'epsilon_decay': 0.9999,
                                                     },

                                     'minibatch_size': 32,
                                     'target_network_update_steps_cadence': 32,
                                     'minibatch_steps_cadence': 4,
                                     },
                            'WCDQN' : {'layers': [64,32],
                                      'layers_subproblems': [64,32],
                                      'learning_rates': {'lr': 0.0001,
                                                         'lr_subproblem': 0.00001,
                                                         },
                                      'replay_buffer': {'transitions_burn_in_size' : 10000,
                                                        'replay_memory_size': 1000000,
                                                        },
                                      'exploration': {'exploration_initial_epsilon' : 1.,
                                                      'exploration_final_epsilon' : 0.05,
                                                        'epsilon_decay': 0.9999,
                                                      },

                                      'minibatch_size': 32,
                                      'target_network_update_steps_cadence' : 32,
                                      'target_network_update_steps_cadence_subproblem' : 32,
                                      'minibatch_steps_cadence' : 4,
                                      'minibatch_steps_cadence_subproblem' : 4,
                                      'num_lambda' : 100,
                                      'lambda_max': 1.5,
                                      'loss_mse_tau' : 1.0,
                                      'loss_ub_tau' : 10,
                                      'loss_tau_subproblem' : 1.5,
                                      },
                            'OTDQN': {'layers': [64,32],
                                     'learning_rates': {'lr': 0.0001,
                                                        },
                                     'replay_buffer': {
                                         'transitions_burn_in_size' : 10000,
                                         'replay_memory_size': 1000000,
                                     },
                                     'exploration': {'exploration_initial_epsilon': 1.,
                                                     'exploration_final_epsilon': 0.05,
                                                     'epsilon_decay': 0.9999,
                                                     },

                                     'minibatch_size': 32,
                                     'target_network_update_steps_cadence': 32,
                                     'minibatch_steps_cadence': 4,
                                     },
                            },

         }

