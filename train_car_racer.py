import torch as nn
import torch.nn
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation
from Dreamer import Dreamer
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import argparse
torch.set_float32_matmul_precision('high')

def main(config): 
    device = torch.device(config['device'])
    dreamer_agent = Dreamer(
        config,
        device=device
    )

    env_id = config['env_id']
    env = gym.make(env_id, continuous=True)
    evaluation_env = gym.make(env_id, continuous=True)
    env = ResizeObservation(env, tuple(config['observation_dims']))
    evaluation_env = ResizeObservation(evaluation_env, tuple(config['observation_dims']))

    WM_loss_list, actor_loss_list, critic_loss_list, evaluation_list = dreamer_agent.train_dreamer(env, evaluation_env)
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, 'agent.pth')
    dreamer_agent.save_trained_Dreamer(save_path)

    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data/')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'training_logs.npz')
    np.savez(
        save_path,
        world_model_loss=np.array(WM_loss_list),
        actor_loss=np.array(actor_loss_list),
        critic_loss=np.array(critic_loss_list),
        rewards=np.array(evaluation_list)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(config)