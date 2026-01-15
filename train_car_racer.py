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
from DreamerUtils import _sanitize_for_save
from Adaptors import CarRacerAdaptor, ActionRepeat, CropObservation
torch.set_float32_matmul_precision('high')

def main(config): 
    device = torch.device(config['device'])
    dreamer_agent = Dreamer(
        config,
        device=device
    )

    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    env_id = config['env_id']
    env = gym.make(env_id, continuous=True)
    evaluation_env = gym.make(env_id, continuous=True)

    env = CropObservation(env)
    evaluation_env = CropObservation(evaluation_env)

    env = ResizeObservation(env, tuple(config['observation_dims']))
    evaluation_env = ResizeObservation(evaluation_env, tuple(config['observation_dims']))

    env = ActionRepeat(CarRacerAdaptor(env), repeat=4)
    evaluation_env = ActionRepeat(CarRacerAdaptor(evaluation_env), repeat=4)

    WM_loss_list, actor_loss_list, critic_loss_list, evaluation_list = dreamer_agent.train_dreamer(env, evaluation_env)
    model_dir = os.environ.get('SM_MODEL_DIR', './models')
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, 'agent.pth')
    dreamer_agent.save_trained_Dreamer(save_path)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(config['device'])
    main(config)