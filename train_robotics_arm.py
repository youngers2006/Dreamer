import os
import argparse
import torch
import gymnasium as gym
from gymnasium.wrappers import AddRenderObservation, ResizeObservation
import numpy as np
import yaml
import panda_gym

# Import Modules
from Utils import _sanitize_for_save
from Core import Dreamer

# Set matrix mutliplication to fp32
torch.set_float32_matmul_precision('high')

def main(config):
    """
    Takes config file as input and runs training loop on Car Racer environment
    """
    # Initialise Dreamer model with config file
    device = torch.device(config['device'])
    dreamer_agent = Dreamer(
        config,
        device=device
    )

    # Create folders for recording training checkpoints
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    # Setup training and evaluation environments
    env_id = config['env_id']

    base_env = gym.make(env_id, render_mode='rgb_array')
    pixel_env = AddRenderObservation(base_env, render_only=True)
    env = ResizeObservation(pixel_env, tuple(config['observation_dims']))

    base_env = gym.make(env_id, render_mode='rgb_array')
    pixel_env = AddRenderObservation(base_env, render_only=True)
    evaluation_env = ResizeObservation(pixel_env, tuple(config['observation_dims']))

    # train dreamer model on environment
    (WM_loss_list, actor_loss_list,
     critic_loss_list, evaluation_list) = dreamer_agent.train_dreamer(env, evaluation_env)

    # save final model weights
    model_dir = os.environ.get('SM_MODEL_DIR', './models')
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, 'agent.pth')
    dreamer_agent.save_trained_Dreamer(save_path)

    # save training logs for later evaluation
    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', './logs')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'training_logs.npz')
    np.savez(
        save_path,
        world_model_loss=_sanitize_for_save(WM_loss_list),
        actor_loss=_sanitize_for_save(actor_loss_list),
        critic_loss=_sanitize_for_save(critic_loss_list),
        rewards=_sanitize_for_save(evaluation_list)
    )

if __name__ == "__main__":
    # load arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # parse config files into main loop
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(config['device'])
    main(config)