import torch
import numpy as np
from .DreamerUtils import symlog_np

class Buffer:
    """Action replay buffer to sample states from and train the world model.

    This class is the action replay buffer, this stores the data from environment 
    interaction to train the world model and anchor imaginatio rollouts, uses circular
    buffer to speed up storage.

    Args:
        buffer_size (int): Total buffer size.
        sequence_length (int): Length of sampled sequences.
        action_size (int): Size of action vector output.
        observation_dims (int): Size of observation.
        device (str, optional): Storage location of the network ('cpu' or 'cuda'). 
            Defaults to 'cpu'.

    Attributes:
        observation_buffer (ndarray[np.uint8]): Store of environment observations. 
        action_buffer (ndarray[np.float32]): Store of environment actions taken. 
        reward_buffer (ndarray[np.float32]): Store of environment reward returns. 
        continue_buffer (ndarray[np.float32]): Store of environment continue flag returns.
        capactity (int): Length of buffer.
        sequence_length (int): Length of sequences sampled from environment.
        next_idx (int): Count of current buffer position.
        size (int): Current size of the buffer.
        device (str): Storage location of the network ('cpu' or 'cuda').
    """
    def __init__(
            self, buffer_size: int, sequence_length: int, action_size: int,
            observation_dims: tuple[int, int], device='cpu'
        ):
        self.observation_buffer = np.zeros(
            (buffer_size, 3, *observation_dims), dtype=np.uint8)
        self.action_buffer = np.zeros(
            (buffer_size, action_size), dtype=np.float32)
        self.reward_buffer = np.zeros(
            (buffer_size, 1), dtype=np.float32)
        self.continue_buffer = np.zeros(
            (buffer_size, 1), dtype=np.float32)

        self.capacity = buffer_size
        self.sequence_length = sequence_length
        self.device = device

        self.next_idx = 0
        self.size = 0

    def add_to_buffer(
            self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, continue_: np.ndarray
        ):
        """Takes inputs and adds them to the replay buffer.

        This function runs a takes inputs, o_t, a_t, r_t, c_t, and adds them to the buffer.

        Args:
            observation (np.ndarray): Observation from environment interaction, 
            uint8 to save memory.
            action (np.ndarray): Action taken in environment.
            reward (np.ndarray): Reward obtained in the environment from taking a_t.
            continue_ (np.ndarray): Flag to signify if the episode has ended.
        """

        # Add recorded data to the buffer in intended format
        self.observation_buffer[self.next_idx] = np.array(observation, dtype=np.uint8)
        self.action_buffer[self.next_idx] = np.array(action, dtype=np.float32)
        self.continue_buffer[self.next_idx] = np.array(continue_, dtype=np.float32)

        # Shift reward into buckets and record it
        reward = np.array(reward, dtype=np.float32)
        reward_symlog = symlog_np(reward)
        self.reward_buffer[self.next_idx] = reward_symlog
    
        # Increase the size count, only if the max length is greater than the buffer size
        self.next_idx = (self.next_idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size = self.size + 1

    def sample_sequences(self, batch_size: int):
        """Samples a batch of sequences from the buffer.

        This function samples a batch of a given size of sequences from
        seen data stored in the buffer.

        Args:
            batch_size (int): Number of sequences to sample from the buffer.
        
        Returns:
            observations (torch.Tensor): Sampled observation array.
            actions (torch.Tensor): Sampled action array.
            rewards (torch.Tensor): Sampled reward array.
            continues (torch.Tensor): Sampled continue array.
            sequence_length (int): Length of sampled sequences.

        Raises:
            ValueError: Not enough data in buffer to sample a full sequence.
        """
        if self.size < self.sequence_length:
            raise ValueError("Not enough data in buffer to sample a full sequence")

        # Define upper bound for where a seq can start
        # Using a check to ensure that start indices dotn wrap around
        if self.size < self.capacity:
            valid_starts_index = self.size - self.sequence_length + 1
        else:
            valid_starts_index = self.capacity - self.sequence_length + 1

        # Create batch of starting indices
        start_indices = np.random.randint(
            0, valid_starts_index, size=batch_size
        )

        # Fix sampling jumps in the sequences by checking that the update pointer
        # isnt in the sample sequence
        if self.size == self.capacity:
            end_indices = start_indices + self.sequence_length
            invalid_mask = (start_indices < self.next_idx) & (self.next_idx < end_indices)
            while np.any(invalid_mask):
                num_invalid = np.sum(invalid_mask)
                new_proposals = np.random.randint(0, self.capacity, size=num_invalid)
                start_indices[invalid_mask] = new_proposals

                end_indices = start_indices + self.sequence_length
                invalid_mask = (start_indices < self.next_idx) & (self.next_idx < end_indices)

        # Build indices array, uses broadcasting to allow addition
        indices = start_indices[:, None] + np.arange(self.sequence_length)[None, :]

        # Sample from the buffer
        observations = self.observation_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        continues = self.continue_buffer[indices]
        sequence_length = self.sequence_length

        # Convert samples to tensors and return them
        observations = torch.tensor(
            observations, dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device)
        continues = torch.tensor(
            continues, dtype=torch.float32, device=self.device)
        return observations, actions, rewards, continues, sequence_length