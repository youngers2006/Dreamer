import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from WorldModel import WorldModel
from Agent import Agent
from Buffer import Buffer
from tqdm import tqdm

class Dreamer(nn.Module):
    def __init__(
            self, 
            hidden_state_dims,
            latent_state_dims,
            observation_dims,
            action_dims,
            world_model_lr,
            world_model_betas,
            world_model_eps,
            WM_epochs, 
            beta_prediction,
            beta_dynamics,
            beta_representation,
            critic_reward_buckets,
            encoder_filter_num_1,
            encoder_filter_num_2,
            encoder_hidden_layer_nodes,
            decoder_filter_num_1,
            decoder_filter_num_2,
            decoder_hidden_layer_nodes,
            dyn_pred_hidden_num_nodes_1,
            dyn_pred_hidden_num_nodes_2,
            rew_pred_hidden_num_nodes_1,
            rew_pred_hidden_num_nodes_2,
            cont_pred_hidden_num_nodes_1,
            cont_pred_hidden_num_nodes_2,
            actor_lr,
            actor_betas,
            actor_eps,
            critic_lr, 
            critic_betas,
            critic_eps,
            AC_epochs, 
            hidden_layer_actor_1_size,
            hidden_layer_actor_2_size,
            hidden_layer_critic_1_size,
            hidden_layer_critic_2_size,
            horizon, 
            batch_size, 
            training_iterations,
            random_iterations,
            nu,
            lambda_,
            gamma,
            buffer_size,
            sequence_length,
            seed, 
            device
        ):
        super().__init__()
        self.hidden_state_dims = hidden_state_dims
        self.action_dims = action_dims
        self.observation_dims = observation_dims
        self.latent_state_dims = latent_state_dims

        self.world_model = WorldModel(
            hidden_state_dims,
            latent_state_dims,
            observation_dims,
            action_dims,
            horizon,
            batch_size,
            world_model_lr,
            world_model_betas,
            world_model_eps,
            beta_prediction,
            beta_dynamics,
            beta_representation,
            encoder_filter_num_1,
            encoder_filter_num_2,
            encoder_hidden_layer_nodes,
            decoder_filter_num_1,
            decoder_filter_num_2,
            decoder_hidden_layer_nodes,
            dyn_pred_hidden_num_nodes_1,
            dyn_pred_hidden_num_nodes_2,
            rew_pred_hidden_num_nodes_1,
            rew_pred_hidden_num_nodes_2,
            critic_reward_buckets,
            cont_pred_hidden_num_nodes_1,
            cont_pred_hidden_num_nodes_2,
            device=device
        ) 
        self.agent = Agent(
            action_dims,
            latent_state_dims,
            hidden_state_dims,
            hidden_layer_actor_1_size,
            hidden_layer_actor_2_size,
            hidden_layer_critic_1_size,
            hidden_layer_critic_2_size,
            critic_reward_buckets,
            actor_lr,
            actor_betas,
            actor_eps,
            critic_lr,
            critic_betas,
            critic_eps,
            nu,
            lambda_,
            gamma,
            device=device
        )
        self.buffer = Buffer(
            buffer_size,
            sequence_length,
            action_dims,
            observation_dims,
            device=device
        )
        self.horizon = horizon
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.training_iterations = training_iterations
        self.random_iterations = random_iterations

        self.hidden_state_dims = hidden_state_dims

        self.WM_epochs = WM_epochs
        self.AC_epochs = AC_epochs
        self.seed = seed
        self.device = device

    def dream_episodes(self, starting_latent_state_batch, starting_hidden_state_batch):
        def dream_episode(starting_latent_state, starting_hidden_state):
            hidden_state = starting_hidden_state.unsqueeze(0)
            latent_state = starting_latent_state.unsqueeze(0)
            hidden_states = []
            latent_states = []
            rewards = []
            actions = []
            continues_ = []
            a_mus = []
            a_sigmas = []
            for _ in range(self.horizon):
                action, a_mu, a_sigma = self.agent.actor.act(hidden_state, latent_state)
                hidden_state_, latent_state_, reward, continue_ = self.world_model.imagine_step(hidden_state, latent_state, action)
                hidden_states.append(hidden_state.squeeze(1)) ; latent_states.append(latent_state.squeeze(1))
                rewards.append(reward) ; actions.append(action) ; continues_.append(continue_)
                a_mus.append(a_mu) ; a_sigmas.append(a_sigma)
                hidden_state = hidden_state_ ; latent_state = latent_state_
            latent_states = torch.stack(latent_states, dim=1)
            hidden_states = torch.stack(hidden_states, dim=1)
            actions = torch.stack(actions, dim=1)
            rewards = torch.stack(rewards, dim=1)
            continues_ = torch.stack(continues_, dim=1)
            a_mus = torch.stack(a_mus, dim=1)
            a_sigmas = torch.stack(a_sigmas, dim=1)
            return (
                latent_states.squeeze(0), 
                hidden_states.squeeze(0), 
                actions.squeeze(0), 
                rewards.squeeze(0), 
                continues_.squeeze(0), 
                a_mus.squeeze(0), 
                a_sigmas.squeeze(0)
            )
    
        dream_batch_fn = torch.vmap(dream_episode, in_dims=(0,0), randomness='different')
        z_batch_seq, h_batch_seq, a_batch_seq, rewards_batch_seq, continues_batch_seq, a_mu_batch_seq, a_sigma_batch_seq = dream_batch_fn(
            starting_latent_state_batch,
            starting_hidden_state_batch
        ) 
        return z_batch_seq, h_batch_seq, a_batch_seq, rewards_batch_seq, continues_batch_seq, a_mu_batch_seq, a_sigma_batch_seq
    
    def rollout_policy(self, env, random_policy=False):
        observation, _ = env.reset(seed=self.seed)
        observation = observation.transpose(2,0,1).astype(np.uint8)
        observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        continue_ = True
        hidden_state = torch.zeros(1, 1, self.hidden_state_dims, dtype=torch.float32, device=self.device)
        latent_state, _ = self.world_model.encoder.encode(hidden_state, observation_tensor)
        latent_state = latent_state.unsqueeze(0).unsqueeze(0)
        for _ in range(self.sequence_length):
            if random_policy:
                action_np = env.action_space.sample()
                action = torch.tensor(action_np, dtype=torch.float32, device=self.device)
                action = action.unsqueeze_(0).unsqueeze(0)
            else:
                action, _, _ = self.agent.actor.act(hidden_state, latent_state)
                action_np = action.detach().cpu().numpy().squeeze()
            observation_, reward, terminated, truncated, _ = env.step(action_np)
            observation_ = observation_.transpose(2,0,1).astype(np.uint8)
            observation__tensor = torch.tensor(observation_, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            done = (terminated or truncated)
            continue_ = (1 - done)
            self.buffer.add_to_buffer(observation, action_np, reward, continue_)
            if done:
                self.seed += 1
                observation, _ = env.reset(seed=self.seed)
                observation = observation.transpose(2,0,1)[np.newaxis, :].astype(np.uint8)
                observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
                continue_ = True
                hidden_state = torch.zeros(1, 1, self.hidden_state_dims)
                latent_state, _ = self.world_model.encoder.encode(hidden_state, observation_tensor)
            else:
                latent_state, hidden_state, _ = self.world_model.observe_step(latent_state, hidden_state, action, observation__tensor)
                observation = observation_
                observation_tensor = observation__tensor

    def train_world_model(self):
        loss_list = []
        for _ in tqdm(range(self.WM_epochs), desc="Training World Model On Buffer Data", leave=False):
            observation_seq_batch, action_seq_batch, reward_seq_batch, continue_seq_batch, _ = self.buffer.sample_sequences(
                batch_size=self.batch_size
            )
            
            loss_world_model = self.world_model.training_step(observation_seq_batch, action_seq_batch, reward_seq_batch, continue_seq_batch)
            loss_list.append(loss_world_model)
        return loss_list

    def warm_start_generator(self, observation_seq_batch, action_seq_batch, sequence_length):
        hidden_batch = torch.zeros(self.batch_size, 1, self.hidden_state_dims, dtype=torch.float32, device=self.device)
        latent_batch, _ = self.world_model.encoder.encode(hidden_batch, observation_seq_batch[:, 0:1, :])
        warmup_length = sequence_length // 2
        for t in range(1, warmup_length):
            latent_batch, hidden_batch, _ = self.world_model.observe_step(
                latent_batch, 
                hidden_batch, 
                action_seq_batch[:,t-1:t,:], 
                observation_seq_batch[:, t:t+1, :]
            )
        return latent_batch, hidden_batch
    
    def train_Agent(self):
        loss_actor_list = [] ; loss_critic_list = []
        for _ in tqdm(range(self.AC_epochs), desc="Training Agent in Dreams", leave=False):
            observation_seq_batch, action_seq_batch, _, _, sequence_length = self.buffer.sample_sequences(batch_size=self.batch_size)
            initial_latent_batch, initial_hidden_batch = self.warm_start_generator(observation_seq_batch, action_seq_batch, sequence_length)
            latent_seq_batch_dream, hidden_seq_batch_dream, action_seq_batch_dream, reward_seq_batch_dream, continue_seq_batch_dream, a_mu_batch_seq, a_sigma_batch_seq = self.dream_episodes(
                initial_latent_batch,
                initial_hidden_batch
            )
            loss_actor, loss_critic = self.agent.train_step(
                latent_seq_batch_dream, 
                hidden_seq_batch_dream, 
                reward_seq_batch_dream, 
                continue_seq_batch_dream, 
                action_seq_batch_dream, 
                a_mu_batch_seq, 
                a_sigma_batch_seq
            )
            loss_actor_list.append(loss_actor)
            loss_critic_list.append(loss_critic)
        loss_actor_list = torch.stack(loss_actor_list, dim=0)
        loss_critic_list = torch.stack(loss_critic_list, dim=0)
        return loss_actor_list.mean(dim=0), loss_critic_list.mean(dim=0)
    
    def load_pretrained_dreamer(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
    
    def save_trained_Dreamer(self, save_path):
        torch.save(self.state_dict(), save_path)

    def evaluate_agent(self, env, eval_episodes):
        reward_list = []
        for _ in tqdm(range(eval_episodes), desc="Evaluating Agent", leave=False):
            self.seed += 1
            total_reward = 0
            observation, _ = env.reset(seed=self.seed)
            observation = observation.transpose(2,0,1).astype(np.uint8)
            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            continue_ = True
            hidden_state = torch.zeros(self.hidden_state_dims, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            latent_state, _ = self.world_model.encoder.encode(hidden_state, observation_tensor)
            while continue_:
                action, _, _ = self.agent.actor.act(hidden_state, latent_state)
                action_np = action.detach().cpu().numpy().squeeze(0).squeeze(0)
                observation_, reward, terminated, truncated, _ = env.step(action_np)
                observation_ = observation_.transpose(2,0,1).astype(np.uint8)
                observation__tensor = torch.tensor(observation_, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                total_reward += reward
                done = (terminated or truncated)
                continue_ = (1 - done)
                latent_state, hidden_state, _ = self.world_model.observe_step(latent_state, hidden_state, action, observation__tensor)
                observation = observation_
                observation_tensor = observation__tensor
            reward_list.append(total_reward)
        reward_list = torch.tensor(reward_list, dtype=torch.float32, device=self.device)
        return reward_list.mean()
    
    def train_dreamer(self, env, eval_env):
        WM_loss_list = [] ; actor_loss_list = [] ; critic_loss_list = []
        evaluation_list = []
        print("Starting Training...")
        print("Starting Random Kickstart.")
        for _ in tqdm(range(self.random_iterations), desc=f"Kickstarting Dreamer Agent.", leave=True):
            self.rollout_policy(env, random_policy=True)
            WM_loss = self.train_world_model()
            actor_loss, critic_loss = self.train_Agent()
            WM_loss_list.append(WM_loss) ; actor_loss_list.append(actor_loss) ; critic_loss_list.append(critic_loss)
        print("Starting Training Loop...")
        eval_reward = self.evaluate_agent(eval_env, eval_episodes=3)
        evaluation_list.append(eval_reward)
        for iter in tqdm(range(self.training_iterations), desc="Training Dreamer Agent.", leave=True):
            self.rollout_policy(env, random_policy=False)
            WM_loss = self.train_world_model()
            actor_loss, critic_loss = self.train_Agent()
            WM_loss_list.append(WM_loss) ; actor_loss_list.append(actor_loss) ; critic_loss_list.append(critic_loss)
            if iter % 5 == 0:
                eval_reward = self.evaluate_agent(eval_env, eval_episodes=3)
                evaluation_list.append(eval_reward)
        print("Training Complete.")
        eval_reward = self.evaluate_agent(eval_env, eval_episodes=10)
        evaluation_list.append(eval_reward)
        return WM_loss_list, actor_loss_list, critic_loss_list, evaluation_list