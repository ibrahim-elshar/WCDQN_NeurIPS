import numpy as np


# Make to stock
###############
num_products = 10
max_resource = 3

config = {'make_to_stock' : {'num_experiments': 5,
                            'Labels': ['DQN','DDQN', 'OTDQN', 'WCDQN'],
                            'environment_name' :'make_to_stock',
                            'seed' : 10,
                            'num_episodes' : 5000, 
                            'episode_limit' : 30, 
                            'curve_episode_cadence': 100,
                            'discount_factor': 0.99,
                            ###
                            'num_products' : num_products,
                            'max_resource' : max_resource,
                            'storage_capacity': np.array([20., 30., 10., 15., 10., 10., 25., 30., 15., 10.]),
                            'max_allowable_backorders' : np.ones(num_products)*5,
                            'lambda_demand' :  np.array([0.3, 0.7, 0.5, 1., 1.4, 0.9, 1.1, 1.2, 0.3, 0.6]),
                            'holding_cost':  np.array([0.1, 0.2, 0.05, 0.3, 0.2, 0.5, 0.3, 0.4, 0.15, 0.12]),
                            'backorder_cost': np.array([3., 1.2, 5.15, 1.3, 1.1, 1.1, 10.3, 1.05, 1., 3.1]), 
                            'lostsales_cost': np.array([30.1, 3.3, 10.05, 3.9, 3.7, 3.6, 40.3, 4.5, 12.55, 44.1]),
                            'production_rate_coefficients': np.array([12., 5.971]),
                            ###
                            'state_dim' : num_products+1,
                            'subproblem_state_dim' : 2,
                            'subproblem_num_actions': np.arange(max_resource+1),
                            ##
                            'DQN' : {'layers': [ 64,32],
                                      'learning_rates': {'lr': 0.0001,
                                                         },  
                                      'replay_buffer': {
                                                        'transitions_burn_in_size' : 10000,
                                                        'replay_memory_size': 1000000,
                                                        },
                                      'exploration': {'exploration_initial_epsilon' : 1.,
                                                      'exploration_final_epsilon' : 0.05,
                                                      'epsilon_decay': 0.9999,
                                                      },
                                                      
                                      'minibatch_size':32,
                                      'target_network_update_steps_cadence' : 32,
                                      'minibatch_steps_cadence' : 4,
                                      },
                            'DDQN': {'layers': [64,32],
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
                                      'learning_rates': {'lr': 0.00001,
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
                                      'target_network_update_steps_cadence_subproblem' : 4000,
                                      'minibatch_steps_cadence' : 4,
                                      'minibatch_steps_cadence_subproblem' : 4,
                                      'num_lambda' : 100,
                                      'lambda_max': 1.5,
                                      'loss_mse_tau' : 1,
                                      'loss_ub_tau' : 10,
                                      'loss_tau_subproblem' : 1.5,
                                      },

                            'OTDQN': {'layers': [64,32],
                                     'learning_rates': {'lr': 0.001,
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
