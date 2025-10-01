import torch
import numpy as np

class Buffer:
    def __init__(self, buffer_size, sequence_length, action_size, observation_dims):
        self.observation_buffer = np.zeros((buffer_size, *observation_dims), dtype=np.uint8)
        self.action_buffer = np.zeros((buffer_size, action_size), dtype=np.float32)
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.continue_buffer = np.zeros((buffer_size, 1), dtype=np.float32)

        self.capacity = buffer_size
        self.sequence_length = sequence_length

        self.next_idx = 0
        self.size = 0

    def add_to_buffer(self, observation, action, reward, continue_):
        self.observation_buffer[self.next_idx] = np.array(observation, dtype=np.uint8)
        self.action_buffer[self.next_idx] = np.array(action, dtype=np.float32)
        self.reward_buffer[self.next_idx] = np.array(reward, dtype=np.float32)
        self.continue_buffer[self.next_idx] = np.array(continue_, dtype=np.float32)

        self.next_idx = (self.next_idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size = self.size + 1

    def sample_sequences(self, batch_size):
        if self.size < self.sequence_length:
            raise ValueError("Not enough data in buffer to sample a full sequence")
        
        valid_starts_index = self.size - self.sequence_length + 1
        start_index = np.random.randint(0, valid_starts_index, size=batch_size)
        indices = start_index[:, None] + np.arange(self.sequence_length)[None, :]

        observations = self.observation_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        continues = self.continue_buffer[indices]
        sequence_length = self.sequence_length

        return observations, actions, rewards, continues, sequence_length