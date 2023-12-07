# code from Optimality Tightening paper 
# Frank S He, Yang Liu, Alexander G Schwing, and Jian Peng. Learning to play in a day: Faster
# deep reinforcement learning by optimality tightening. arXiv preprint arXiv:1611.01606, 2016.
import numpy as np
import time


class OptimalityTighteningReplayMemory(object):
    def __init__(self, state_size, rng, max_steps=1000000, phi_length=1, discount=0.99, batch_size=32,
                 transitions_len=4):
        self.max_steps = max_steps
        self.phi_length = phi_length
        self.rng = rng
        self.discount = discount
        self.discount_table = np.power(self.discount, np.arange(30)) 

        self.state_size = state_size
        self.states = np.zeros((max_steps, self.state_size), dtype='float32')
        self.actions = np.zeros(max_steps, dtype='int32')
        self.rewards = np.zeros(max_steps, dtype='float32')
        self.return_value = np.zeros(max_steps, dtype='float32')
        self.terminal = np.zeros(max_steps, dtype='bool')
        self.terminal_index = np.zeros(max_steps, dtype='int32')
        self.start_index = np.zeros(max_steps, dtype='int32')

        self.bottom = 0
        self.top = 0
        self.size = 0

        self.center_states = np.zeros((batch_size,
                                     self.phi_length,
                                     self.state_size),
                                     dtype='float32')
        self.forward_states = np.zeros((batch_size,
                                      transitions_len,
                                      self.phi_length,
                                      self.state_size),
                                     dtype='float32')
        self.backward_states = np.zeros((batch_size,
                                       transitions_len,
                                       self.phi_length,
                                       self.state_size),
                                      dtype='float32')

        self.center_positions = np.zeros((batch_size, 1), dtype='int32')
        self.forward_positions = np.zeros((batch_size, transitions_len), dtype='int32')
        self.backward_positions = np.zeros((batch_size, transitions_len), dtype='int32')

        self.center_actions = np.zeros((batch_size, 1), dtype='int32')
        self.backward_actions = np.zeros((batch_size, transitions_len), dtype='int32')

        self.center_terminals = np.zeros((batch_size, 1), dtype='bool')
        self.center_rewards = np.zeros((batch_size, 1), dtype='float32')

        self.center_return_values = np.zeros((batch_size, 1), dtype='float32')
        self.forward_return_values = np.zeros((batch_size, transitions_len), dtype='float32')
        self.backward_return_values = np.zeros((batch_size, transitions_len), dtype='float32')

        self.forward_discounts = np.zeros((batch_size, transitions_len), dtype='float32')
        self.backward_discounts = np.zeros((batch_size, transitions_len), dtype='float32')

    def add_sample(self, state, action, reward, terminal, return_value=0.0, start_index=-1):

        self.states[self.top] = state
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal
        self.return_value[self.top] = return_value
        self.start_index[self.top] = start_index
        self.terminal_index[self.top] = -1

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        return self.size

    def last_phi(self):
        indexes = np.arange(self.top - self.phi_length, self.top)
        return self.states.take(indexes, axis=0, mode='wrap')

    def phi(self, state):
        indexes = np.arange(self.top - self.phi_length + 1, self.top)

        phi = np.empty((self.phi_length, self.state_size), dtype='float32')
        phi[0:self.phi_length - 1] = self.states.take(indexes,
                                                    axis=0,
                                                    mode='wrap')
        phi[-1] = state
        return phi

    def random_close_transitions_batch(self, batch_size, transitions_len):
        transition_range = transitions_len
        count = 0
        while count < batch_size:
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - self.phi_length)

            all_indices = np.arange(index, index + self.phi_length)
            center_index = index + self.phi_length - 1

            if np.any(self.terminal.take(all_indices[0:-1], mode='wrap')):
                continue
            if np.any(self.terminal_index.take(all_indices, mode='wrap') == -1):
                continue
            terminal_index = self.terminal_index.take(center_index, mode='wrap')
            start_index = self.start_index.take(center_index, mode='wrap')
            self.center_positions[count] = center_index
            self.center_terminals[count] = self.terminal.take(center_index, mode='wrap')
            self.center_rewards[count] = self.rewards.take(center_index, mode='wrap')

            """ get forward transitions """
            if terminal_index < center_index:
                terminal_index += self.size
            max_forward_index = max(min(center_index + transition_range, terminal_index), center_index+1) + 1
            self.forward_positions[count] = center_index + 1
            for i, j in zip(range(transitions_len), range(center_index + 1, max_forward_index)):
                self.forward_positions[count, i] = j
            """ get backward transitions """
            if start_index + self.size < center_index:
                start_index += self.size
            min_backward_index = max(center_index - transition_range, start_index+self.phi_length-1)
            self.backward_positions[count] = center_index + 1
            for i, j in zip(range(transitions_len), range(center_index - 1, min_backward_index - 1, -1)):
                self.backward_positions[count, i] = j
                if self.terminal_index.take(j, mode='wrap') == -1:
                    self.backward_positions[count, i] = center_index + 1

            self.center_states[count] = self.states.take(all_indices, axis=0, mode='wrap')
            for j in range(transitions_len):
                forward_index = self.forward_positions[count, j]
                backward_index = self.backward_positions[count, j]
                self.forward_states[count, j] = self.states.take(
                    np.arange(forward_index - self.phi_length + 1, forward_index + 1), axis=0, mode='wrap')
                self.backward_states[count, j] = self.states.take(
                    np.arange(backward_index - self.phi_length + 1, backward_index + 1), axis=0, mode='wrap')
            self.center_actions[count] = self.actions.take(center_index, mode='wrap')
            self.backward_actions[count] = self.actions.take(self.backward_positions[count], mode='wrap')
            self.center_return_values[count] = self.return_value.take(center_index, mode='wrap')
            self.forward_return_values[count] = self.return_value.take(self.forward_positions[count], mode='wrap')
            self.backward_return_values[count] = self.return_value.take(self.backward_positions[count], mode='wrap')
            distance = np.absolute(self.forward_positions[count] - center_index)
            self.forward_discounts[count] = self.discount_table[distance]
            distance = np.absolute(self.backward_positions[count] - center_index)
            self.backward_discounts[count] = self.discount_table[distance]
            count += 1

    def random_transitions_batch(self, batch_size, transitions_len, transition_range=10):
        count = 0
        while count < batch_size:
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - self.phi_length)

            all_indices = np.arange(index, index + self.phi_length)
            center_index = index + self.phi_length - 1
            if np.any(self.terminal.take(all_indices[0:-1], mode='wrap')):
                continue
            if np.any(self.terminal_index.take(all_indices, mode='wrap') == -1):
                continue
            terminal_index = self.terminal_index.take(center_index, mode='wrap')
            start_index = self.start_index.take(center_index, mode='wrap')
            self.center_positions[count] = center_index
            self.center_terminals[count] = self.terminal.take(center_index, mode='wrap')
            self.center_rewards[count] = self.rewards.take(center_index, mode='wrap')

            """ get forward transitions """
            if terminal_index < center_index:
                terminal_index += self.size
            max_forward_index = max(min(center_index + transition_range, terminal_index), center_index+1) + 1
            self.forward_positions[count, 0] = center_index+1
            if center_index + 2 >= max_forward_index:
                self.forward_positions[count, 1:] = center_index + 1
            else:
                self.forward_positions[count, 1:] = self.rng.randint(center_index+2, max_forward_index, transitions_len-1)
            """ get backward transitions """

            if start_index + self.size < center_index:
                start_index += self.size
            min_backward_index = max(center_index - transition_range, start_index+self.phi_length-1)
            if min_backward_index >= center_index:
                self.backward_positions[count] = [center_index + 1] * transitions_len
            else:
                if center_index > self.top > min_backward_index:
                    min_backward_index = self.top
                self.backward_positions[count] = self.rng.randint(min_backward_index, center_index, transitions_len)

            self.center_states[count] = self.states.take(all_indices, axis=0, mode='wrap')
            for j in range(transitions_len):
                forward_index = self.forward_positions[count, j]
                backward_index = self.backward_positions[count, j]
                self.forward_states[count, j] = self.states.take(
                    np.arange(forward_index - self.phi_length + 1, forward_index + 1), axis=0, mode='wrap')
                self.backward_states[count, j] = self.states.take(
                    np.arange(backward_index - self.phi_length + 1, backward_index + 1), axis=0, mode='wrap')
            self.center_actions[count] = self.actions.take(center_index, mode='wrap')
            self.backward_actions[count] = self.actions.take(self.backward_positions[count], mode='wrap')
            self.center_return_values[count] = self.return_value.take(center_index, mode='wrap')
            self.forward_return_values[count] = self.return_value.take(self.forward_positions[count], mode='wrap')
            self.backward_return_values[count] = self.return_value.take(self.backward_positions[count], mode='wrap')
            distance = np.absolute(self.forward_positions[count] - center_index)
            self.forward_discounts[count] = self.discount_table[distance]
            distance = np.absolute(self.backward_positions[count] - center_index)
            self.backward_discounts[count] = self.discount_table[distance]
            count += 1

    def random_states(self, size):
        states = np.zeros((size,
                         self.phi_length + 1,
                         self.state_size,
                         ),
                        dtype='uint8')

        count = 0
        while count < size:
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - self.phi_length)
            all_indices = np.arange(index, index + self.phi_length + 1)
            end_index = index + self.phi_length - 1
            if np.any(self.terminal.take(all_indices[0:-2], mode='wrap')):
                continue
            states[count] = self.states.take(all_indices, axis=0, mode='wrap')
            count += 1
        return states

