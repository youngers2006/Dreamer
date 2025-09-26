import torch
class Buffer:
    def __init__(self, buffer_size, action_size, observation_num_rows, observation_num_columns, device='cpu'):
        observation_buffer = torch.zeros(buffer_size, observation_num_rows, observation_num_columns, dtype=torch.float32, device=device)
        action_buffer = torch.zeros(buffer_size, action_size, dtype=torch.float32, device=device)
        reward_buffer = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        continue_buffer = torch.zeros(buffer_size, dtype=torch.float32, device=device)

    def sample_sequence(self):
