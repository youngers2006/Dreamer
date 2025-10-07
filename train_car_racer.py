import torch as nn
import torch.nn
import gymnasium as gym
from Dreamer import Dreamer
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import argparse
import csv

def main(config): 
    device = torch.device(config['device'])
    dreamer_agent = Dreamer(
        hidden_state_dims=config['hidden_state_dims'],
        latent_state_dims=tuple(config['latent_state_dims']),
        observation_dims=tuple(config['observation_dims']),
        action_dims=config['action_dims'],
        world_model_lr=config['world_model_lr'],
        world_model_betas=tuple(config['world_model_betas']),
        world_model_eps=config['world_model_eps'],
        WM_epochs=config['WM_epochs'],
        beta_prediction=config['beta_prediction'],
        beta_dynamics=config['beta_dynamics'],
        beta_representation=config['beta_representation'],
        critic_reward_buckets=config['critic_reward_buckets'],
        encoder_filter_num_1=config['encoder_filter_num_1'],
        encoder_filter_num_2=config['encoder_filter_num_2'],
        encoder_hidden_layer_nodes=config['encoder_hidden_layer_nodes'],
        decoder_filter_num_1=config['decoder_filter_num_1'],
        decoder_filter_num_2=config['decoder_filter_num_2'],
        decoder_hidden_layer_nodes=config['decoder_hidden_layer_nodes'],
        dyn_pred_hidden_num_nodes_1=config['dyn_pred_hidden_num_nodes_1'],
        dyn_pred_hidden_num_nodes_2=config['dyn_pred_hidden_num_nodes_2'],
        rew_pred_hidden_num_nodes_1=config['rew_pred_hidden_num_nodes_1'],
        rew_pred_hidden_num_nodes_2=config['rew_pred_hidden_num_nodes_2'],
        cont_pred_hidden_num_nodes_1=config['cont_pred_hidden_num_nodes_1'],
        cont_pred_hidden_num_nodes_2=config['cont_pred_hidden_num_nodes_2'],
        actor_lr=config['actor_lr'],
        actor_betas=tuple(config['actor_betas']),
        actor_eps=config['actor_eps'],
        critic_lr=config['critic_lr'],
        critic_betas=tuple(config['critic_betas']),
        critic_eps=config['critic_eps'],
        AC_epochs=config['AC_epochs'],
        hidden_layer_actor_1_size=config['hidden_layer_actor_1_size'],
        hidden_layer_actor_2_size=config['hidden_layer_actor_2_size'],
        hidden_layer_critic_1_size=config['hidden_layer_critic_1_size'],
        hidden_layer_critic_2_size=config['hidden_layer_critic_2_size'],
        horizon=config['horizon'],
        batch_size=config['batch_size'],
        training_iterations=config['training_iterations'],
        random_iterations=config['random_iterations'],
        nu=config['nu'],
        lambda_=config['lambda_'],
        gamma=config['gamma'],
        buffer_size=config['buffer_size'],
        sequence_length=config['sequence_length'],
        seed=config['seed'],
        device=device
    )

    env_id = config['env_id']
    env = gym.make(env_id, continuous=True)
    evaluation_env = gym.make(env_id, continuous=True)
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