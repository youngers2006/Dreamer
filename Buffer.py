import torch
import numpy as np
from DreamerUtils import symlog_np

class Buffer:
    def __init__(self, buffer_size, sequence_length, action_size, observation_dims, device='cpu'):
        self.observation_buffer = np.zeros((buffer_size, 3, *observation_dims), dtype=np.uint8)
        self.action_buffer = np.zeros((buffer_size, action_size), dtype=np.float32)
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.continue_buffer = np.zeros((buffer_size, 1), dtype=np.float32)

        self.capacity = buffer_size
        self.sequence_length = sequence_length
        self.device = device

        self.next_idx = 0
        self.size = 0

    def add_to_buffer(self, observation, action, reward, continue_):
        self.observation_buffer[self.next_idx] = np.array(observation, dtype=np.uint8)
        self.action_buffer[self.next_idx] = np.array(action, dtype=np.float32)
        self.continue_buffer[self.next_idx] = np.array(continue_, dtype=np.float32)

        reward = np.array(reward, dtype=np.float32)
        reward_symlog = symlog_np(reward)
        self.reward_buffer[self.next_idx] = reward_symlog

        self.next_idx = (self.next_idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size = self.size + 1

    def sample_sequences(self, batch_size):
        if self.size < self.sequence_length:
            raise ValueError("Not enough data in buffer to sample a full sequence")
        
        valid_starts_index = self.size - self.sequence_length + 1
        start_index = np.random.randint(0, valid_starts_index, size=batch_size)
        
        if self.size == self.capacity:
            valid_indices = []
            for idx in start_index:
                end_idx = idx + self.sequence_length
                if idx < self.next_idx < end_idx:
                    new_idx = np.random.randint(0, valid_starts_index) 
                    valid_indices.append(new_idx) 
                else:
                    valid_indices.append(idx)
            start_index = np.array(valid_indices)
        indices = start_index[:, None] + np.arange(self.sequence_length)[None, :]
        indices = indices % self.capacity

        observations = self.observation_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        continues = self.continue_buffer[indices]
        sequence_length = self.sequence_length

        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        continues = torch.tensor(continues, dtype=torch.float32, device=self.device)

        return observations, actions, rewards, continues, sequence_length