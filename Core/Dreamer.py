import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# file module imports
from Utils import _sanitize_for_save, Buffer
from .WorldModel import WorldModel
from .Agent import Agent

class Dreamer(nn.Module):
    """Dreamer agent class.

    This class is the DreamerV3 agent class, it consists of an Encoder-Decoder pair,
    an RSSM, an Actor-Critic REINFORCE agent and a Buffer. The training loop and inference
    functions are all contained in this class.

    Args:
        config (dict): Dictionary storing configurations for the agent.
        device (str, optional): Storage location of the network ('cpu' or 'cuda'). 
            Defaults to 'cpu'.

    Attributes:
        hidden_state_dims (int): Hidden state vector size.
        action_dims (int): Action vector size.
        observation_dims (tuple(int, int)): Observation size.
        latent_state_dims (tuple(int, int)): Latent state matrix size.
        agent_obs (None / torch.Tensor): Intermediate observation store for rollout.
        agent_hidden (None / torch.Tensor): Intermediate hidden state store for rollout.
        agent_latent (None / torch.Tensor): Intermediate latent state store for rollout.
        WM_epochs (int): Number of training epochs for the world model training.
        AC_epochs (int): Number of training epochs for the agent training.
        seed (int): Randomisation seed for the environment.
        device (str): Storage location of the network ('cpu' or 'cuda').
    """
    def __init__(
            self,
            config,
            device
        ):
        super().__init__()

        # Unload config file
        hidden_state_dims=config['hidden_state_dims']
        latent_state_dims=tuple(config['latent_state_dims'])
        observation_dims=tuple(config['observation_dims'])
        action_dims=config['action_dims']
        world_model_lr=config['world_model_lr']
        world_model_betas=tuple(config['world_model_betas'])
        world_model_eps=config['world_model_eps']
        WM_epochs=config['WM_epochs']
        beta_prediction=config['beta_prediction']
        beta_dynamics=config['beta_dynamics']
        beta_representation=config['beta_representation']
        critic_reward_buckets=config['critic_reward_buckets']
        encoder_filter_num_1=config['encoder_filter_num_1']
        encoder_filter_num_2=config['encoder_filter_num_2']
        encoder_hidden_layer_nodes=config['encoder_hidden_layer_nodes']
        decoder_filter_num_1=config['decoder_filter_num_1']
        decoder_filter_num_2=config['decoder_filter_num_2']
        decoder_hidden_layer_nodes=config['decoder_hidden_layer_nodes']
        dyn_pred_hidden_num_nodes_1=config['dyn_pred_hidden_num_nodes_1']
        dyn_pred_hidden_num_nodes_2=config['dyn_pred_hidden_num_nodes_2']
        rew_pred_hidden_num_nodes_1=config['rew_pred_hidden_num_nodes_1']
        rew_pred_hidden_num_nodes_2=config['rew_pred_hidden_num_nodes_2']
        cont_pred_hidden_num_nodes_1=config['cont_pred_hidden_num_nodes_1']
        cont_pred_hidden_num_nodes_2=config['cont_pred_hidden_num_nodes_2']
        actor_lr=config['actor_lr']
        actor_betas=tuple(config['actor_betas'])
        actor_eps=config['actor_eps']
        critic_lr=config['critic_lr']
        critic_betas=tuple(config['critic_betas'])
        critic_eps=config['critic_eps']
        AC_epochs=config['AC_epochs']
        hidden_layer_actor_1_size=config['hidden_layer_actor_1_size']
        hidden_layer_actor_2_size=config['hidden_layer_actor_2_size']
        hidden_layer_critic_1_size=config['hidden_layer_critic_1_size']
        hidden_layer_critic_2_size=config['hidden_layer_critic_2_size']
        horizon=config['horizon']
        batch_size=config['batch_size']
        training_iterations=config['training_iterations']
        random_iterations=config['random_iterations']
        nu=config['nu']
        lambda_=config['lambda_']
        gamma=config['gamma']
        buffer_size=config['buffer_size']
        sequence_length=config['sequence_length']
        seed=config['seed']

        self.hidden_state_dims = hidden_state_dims
        self.action_dims = action_dims
        self.observation_dims = observation_dims
        self.latent_state_dims = latent_state_dims

        self.world_model = WorldModel(
            hidden_state_dims, latent_state_dims, observation_dims,
            action_dims, horizon, batch_size,
            world_model_lr, world_model_betas, world_model_eps,
            beta_prediction, beta_dynamics, beta_representation,
            encoder_filter_num_1, encoder_filter_num_2, encoder_hidden_layer_nodes,
            decoder_filter_num_1, decoder_filter_num_2, decoder_hidden_layer_nodes,
            dyn_pred_hidden_num_nodes_1, dyn_pred_hidden_num_nodes_2, rew_pred_hidden_num_nodes_1,
            rew_pred_hidden_num_nodes_2, critic_reward_buckets, cont_pred_hidden_num_nodes_1,
            cont_pred_hidden_num_nodes_2,
            device=device
        )
        self.agent = Agent(
            action_dims, latent_state_dims, hidden_state_dims,
            hidden_layer_actor_1_size, hidden_layer_actor_2_size, hidden_layer_critic_1_size,
            hidden_layer_critic_2_size, critic_reward_buckets, actor_lr,
            actor_betas, actor_eps, critic_lr, critic_betas,
            critic_eps, nu, lambda_, gamma,
            device=device
        )
        self.buffer = Buffer(
            buffer_size, sequence_length,
            action_dims, observation_dims,
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

        self.agent_obs = None
        self.agent_hidden = None
        self.agent_latent = None

    def dream_episodes(
            self, starting_latent_state_batch: torch.Tensor,
            starting_hidden_state_batch: torch.Tensor
        ):
        """Imagines a sequence anchors at a buffer sample.
        
        Given a starting state use the world model to imagine future 
        trajectories within the horizon.

        Args:
            starting_latent_state_batch (torch.Tensor): Sampled initial 
            batch of latent states to anchor dreamed episodes.
            starting_hidden_state_batch (torch.Tensor): Initial batch of 
            hidden states to anchor dreamed episodes. 
        
        Returns:
            latent_states (torch.Tensor): Batch of imagined latent state sequences.
            hidden_states (torch.Tensor): Batch of imagined hidden state sequences.
            actions (torch.Tensor): Batch of imagined imagined sequences.
            rewards (torch.Tensor): Batch of imagined reward sequences.
            continues_ (torch.Tensor): Batch of imagined continue sequences.
            a_mus (torch.Tensor): Batch of imagined action prediction mean sequences.
            a_sigmas (torch.Tensor): Batch of imagined action prediction std.dev sequences.
        """
        # Initialise latent and hidden states
        hidden_state_batch = starting_hidden_state_batch
        latent_state_batch = starting_latent_state_batch

        # Initialise returns
        hidden_states = []
        latent_states = []
        rewards = []
        actions = []
        continues_ = []
        a_mus = []
        a_sigmas = []

        # Run forward in time
        for _ in range(self.horizon):
            # Run policy on imagined state
            action_batch, a_mu_batch, a_sigma_batch = self.agent.actor.act(
                hidden_state_batch, latent_state_batch, deterministic=False
            )

            # Take action in imagination
            (hidden_state__batch, latent_state__batch, reward_batch,
             continue_batch) = self.world_model.imagine_step(
                hidden_state_batch, latent_state_batch, action_batch
            )

            # Append new state batch to sequence
            hidden_states.append(hidden_state_batch)
            latent_states.append(latent_state_batch)
            rewards.append(reward_batch)
            actions.append(action_batch)
            continues_.append(continue_batch)
            a_mus.append(a_mu_batch)
            a_sigmas.append(a_sigma_batch)

            # Reset state
            hidden_state_batch = hidden_state__batch
            latent_state_batch = latent_state__batch

        # append final state batch to list
        hidden_states.append(hidden_state_batch)
        latent_states.append(latent_state_batch)

        # Concatenate timesteps in sequence dimension
        latent_states = torch.cat(
            latent_states, dim=1)
        hidden_states = torch.cat(
            hidden_states, dim=1)
        actions = torch.cat(
            actions, dim=1)
        rewards = torch.cat(
            rewards, dim=1)
        continues_ = torch.cat(
            continues_, dim=1)
        a_mus = torch.cat(
            a_mus, dim=1)
        a_sigmas = torch.cat(
            a_sigmas, dim=1)
        return latent_states, hidden_states, actions, rewards, continues_, a_mus, a_sigmas

    def rollout_policy(self, env, random_policy=False):
        """Rolls out policy in environment to collect trajectories for the buffer.
        
        Rolls out either the current policy or a random policy on the environment
        to collect data to store in the buffer.

        Args:
            env: Environment to roll out policy on, must respond to gymnasium commands.
            random_policy (bool): Use current policy or random policy.
        """
        # Stop gradient tracking
        with torch.no_grad():

            # If no observation stored then get one by resetting the environment
            if self.agent_obs is None:
                # Reset environment
                observation, _ = env.reset(seed=self.seed)

                # Normalise observation and store as a uint8
                observation = observation.transpose(2,0,1).astype(np.uint8)
                self.agent_obs = (observation.astype(np.float32) / 255.0) - 0.5

                # Initialise hidden as zeros and convert obs to a tensor
                self.agent_hidden = torch.zeros(
                    1, 1, self.hidden_state_dims, dtype=torch.float32, device=self.device
                )
                observation_tensor = torch.tensor(
                    self.agent_obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0).unsqueeze(0)

                # Get latent state from the encoder
                self.agent_latent, _ = self.world_model.encoder.encode(
                    self.agent_hidden, observation_tensor
                )

            # Loop forward in time
            for _ in range(self.sequence_length):
                # Rollout specified policy to produce an action
                if random_policy:
                    action_np = env.action_space.sample()
                    action = torch.tensor(
                        action_np, dtype=torch.float32, device=self.device
                    )
                    action = action.unsqueeze_(0).unsqueeze(0)
                else:
                    action, _, _ = self.agent.actor.act(
                        self.agent_hidden, self.agent_latent, deterministic=False
                    )
                    action_np = action.detach().cpu().numpy().reshape(-1)

                # Step forward in environment
                observation_, reward, terminated, truncated, _ = env.step(action_np)

                # Convert observation to correct datatype, normalise then convert to a tensor
                observation_ = observation_.transpose(2,0,1).astype(np.uint8)
                obs__normalised = (observation_.astype(np.float32) / 255.0) - 0.5
                observation__tensor = torch.tensor(
                    obs__normalised, dtype=torch.float32, device=self.device
                ).unsqueeze(0).unsqueeze(0)

                # Convert done marker to inverse, continue
                done = (terminated or truncated)
                continue_ = (1 - done)

                # Convert observation to correct datatype and normalised
                current_obs_uint8 = ((self.agent_obs + 0.5) * 255.0).astype(np.uint8)

                # Add data to buffer
                self.buffer.add_to_buffer(
                    current_obs_uint8, action_np, reward, continue_
                )

                # If done flag raised
                if done:
                    # Chnage env seed and reset env
                    self.seed += 1
                    observation, _ = env.reset(seed=self.seed)

                    # Manipulate observation and convert to tensor
                    observation = observation.transpose(2,0,1).astype(np.uint8)
                    self.agent_obs = (observation.astype(np.float32) / 255.0) - 0.5
                    observation_tensor = torch.tensor(
                        self.agent_obs, dtype=torch.float32, device=self.device
                    ).unsqueeze(0).unsqueeze(0)

                    # Reset continue flag and hidden state
                    continue_ = True
                    self.agent_hidden = torch.zeros(
                        1, 1, self.hidden_state_dims, dtype=torch.float32, device=self.device
                    )

                    # Produce new initial latent state from initial obs and hidden state
                    self.agent_latent, _ = self.world_model.encoder.encode(
                        self.agent_hidden, observation_tensor
                    )
                else: # If not done
                    # Process data in world model to get latent and hidden state
                    self.agent_obs = obs__normalised
                    self.agent_latent, self.agent_hidden, _ = self.world_model.observe_step(
                        self.agent_latent, self.agent_hidden, action, observation__tensor
                    )

    def train_world_model(self):
        """Trains the world model on sampled data.
        
        Samples a batch of trajectories from the buffer and 
        uses these to train the world model.

        Returns:
            loss_list (list): Training losses for training log.
        """
        # Initialise losses
        loss_list = []

        # Loop through epochs
        for _ in tqdm(range(self.WM_epochs), desc="Training World Model On Buffer Data", leave=False):
            # Sample batch of sequences from the buffer
            (observation_seq_batch, action_seq_batch, reward_seq_batch,
             continue_seq_batch, _) = self.buffer.sample_sequences(batch_size=self.batch_size)

            # Run train step with sampled data
            loss_world_model = self.world_model.training_step(
                observation_seq_batch, action_seq_batch, reward_seq_batch, continue_seq_batch
            )

            # Record loss
            loss_list.append(loss_world_model)
        return loss_list

    def warm_start_generator(
            self, observation_seq_batch: torch.Tensor,
            action_seq_batch: torch.Tensor, sequence_length: int
        ):
        """Generates initial hidden state and latent state for agent training process.
        
        The training process for the agent requires starting imagined trajectories
        at sampled states in the buffer, using a warmup length of half
        the sequence length.

        Args:
            observation_seq_batch (torch.Tensor): Batch of sampled sequences of observations.
            action_seq_batch (torch.Tensor): Batch of sampled sequences of actions.
            sequence_length (int): Length of sampled sequence.

        Returns:
            latent_batch (torch.Tensor): Latent state to warm start agent training.
            hidden_batch (torch.Tensor): Hidden state to warm start agent training.
        """
        # Normalise observations
        observation_seq_batch = (observation_seq_batch.float() / 255.0) - 0.5

        # Create initial hidden batch
        hidden_batch = torch.zeros(
            self.batch_size, 1, self.hidden_state_dims, dtype=torch.float32, device=self.device
        )

        # Create initial latent state using encoder
        latent_batch, _ = self.world_model.encoder.encode(
            hidden_batch, observation_seq_batch[:, 0:1, :]
        )

        # Use half the sampled data to warm start the RSSM
        warmup_length = sequence_length // 2
        for t in range(1, warmup_length):
            # Run RSSM across the sampled states to warmup the sequence model.
            latent_batch, hidden_batch, _ = self.world_model.observe_step(
                latent_batch, hidden_batch, action_seq_batch[:,t-1:t,:],
                observation_seq_batch[:, t:t+1, :]
            )
        return latent_batch, hidden_batch

    def train_Agent(self):
        """Generates initial hidden state and latent state for agent training process.
        
        The training process for the agent requires starting imagined trajectories
        at sampled states in the buffer, using a warmup length of half
        the sequence length.

        Returns:
            mean_actor_loss (torch.Tensor): Mean loss across batch dimension for the actor.
            mean_critic_loss (torch.Tensor): Mean loss across batch dimension for the critic.
        """
        # Initialise loss logs
        loss_actor_list = []
        loss_critic_list = []

        # Loop through training epochs
        for _ in tqdm(range(self.AC_epochs), desc="Training Agent in Dreams", leave=False):
            # Sample batch of sequences from buffer
            (observation_seq_batch, action_seq_batch, _, _, 
             sequence_length) = self.buffer.sample_sequences(
                batch_size=self.batch_size
            )
            
            # Warm start RSSM to produce batch of initial latent and hidden states
            initial_latent_batch, initial_hidden_batch = self.warm_start_generator(
                observation_seq_batch, action_seq_batch, sequence_length
            )

            # Generate batch of episodes in dreams
            (latent_seq_batch_dream, hidden_seq_batch_dream, action_seq_batch_dream,
                reward_seq_batch_dream, continue_seq_batch_dream,
                a_mu_batch_seq, a_sigma_batch_seq) = self.dream_episodes(
                initial_latent_batch, initial_hidden_batch
            )

            # Run agent train step
            loss_actor, loss_critic = self.agent.train_step(
                latent_seq_batch_dream, hidden_seq_batch_dream,
                reward_seq_batch_dream, continue_seq_batch_dream,
                action_seq_batch_dream, a_mu_batch_seq,
                a_sigma_batch_seq
            )

            # Record training losses
            loss_actor_list.append(loss_actor)
            loss_critic_list.append(loss_critic)
        
        # Stack losses into tensor and return mean losses
        loss_actor_list = torch.stack(loss_actor_list, dim=0)
        loss_critic_list = torch.stack(loss_critic_list, dim=0)
        return loss_actor_list.mean(dim=0), loss_critic_list.mean(dim=0)

    def load_pretrained_dreamer(self, path: str):
        """Loads agent from given path.

        Args:
            path (str): Path to load trained agent from.
        """
        self.load_state_dict(
            torch.load(path, weights_only=True)
        )

    def save_trained_Dreamer(self, save_path: str):
        """Saves agent to given path.

        Args:
            path (str): Path to save trained agent to.
        """
        torch.save(
            self.state_dict(), save_path
        )

    def evaluate_agent(self, env, eval_episodes: int):
        """Evaluates current agent performance by testing on the envirnment.

        Args:
            env: Environment to roll out policy on, must respond to gymnasium commands.
            eval_episodes (int): Number of episodes to evaluate agent on.

        Returns:
            average_reward (torch.Tensor): Mean reward from agent testing.
        """
        # Initialise reward log
        reward_list = []

        # Loop through episodes
        for _ in tqdm(range(eval_episodes), desc="Evaluating Agent", leave=False):
            # Change seed and initialise episode reward log
            self.seed += 1
            ep_reward = self.Run(
                env, self.seed, render=False)
            reward_list.append(ep_reward)

        # Convert reward log to a tensor and return it
        reward_list = torch.tensor(
            reward_list, dtype=torch.float32, device=self.device
        )
        return reward_list.mean()

    def train_dreamer(self, env, eval_env):
        """Rolls out policy in environment to collect trajectories for the buffer.
        
        Rolls out either the current policy or a random policy on the environment
        to collect data to store in the buffer.

        Args:
            env: Environment to roll out policy on, must respond to gymnasium commands.
            eval_env (bool): Environment to evaluate policy on, must respond to gymnasium commands.
        """
        # Initialise training logs
        WM_loss_list, actor_loss_list = [], []
        critic_loss_list, evaluation_list = [], []

        # Rollout random policy to collect training data for the buffer and train world model on it
        print("Starting Training...")
        print("Starting Random Kickstart.")
        for iter in tqdm(range(self.random_iterations), desc="Kickstarting Dreamer Agent.", leave=True):
            self.rollout_policy(env, random_policy=True)
            WM_loss = self.train_world_model()
            WM_loss_list.append([x.detach().cpu().item() for x in WM_loss])

        # Evaluate Current policy performance
        eval_reward = self.evaluate_agent(
            eval_env, eval_episodes=3
        )
        evaluation_list.append(
            eval_reward.detach().cpu().item()
        )

        # Run main training loop
        print("Starting Training Loop...")
        for iter in tqdm(range(self.training_iterations), desc="Training Dreamer Agent.", leave=True):
            # Collect training data with policy
            self.rollout_policy(
                env, random_policy=False
            )

            # Train WM and agent
            WM_loss = self.train_world_model()
            actor_loss, critic_loss = self.train_Agent()

            WM_loss_list.append(
                [x.detach().cpu().item() for x in WM_loss])
            actor_loss_list.append(
                actor_loss.detach().cpu().item())
            critic_loss_list.append(
                critic_loss.detach().cpu().item())

            # Save every 1000 iterations
            if iter % 1000 == 0:
                # Save Model
                save_path = os.path.join('./models', f'agent_checkpoint_{iter}.pth')
                self.save_trained_Dreamer(save_path)
            
                # Save Latest Model (overwrite)
                latest_path = os.path.join('./models', 'agent_latest.pth')
                self.save_trained_Dreamer(latest_path)

                # Save Logs
                log_path = os.path.join('./models', 'training_logs.npz')
                np.savez(
                    log_path,
                    world_model_loss=_sanitize_for_save(WM_loss_list),
                    actor_loss=_sanitize_for_save(actor_loss_list),
                    critic_loss=_sanitize_for_save(critic_loss_list),
                    rewards=_sanitize_for_save(evaluation_list)
                )

            # Intermittantly evaluate the agent during training
            if iter % 1000 == 0:
                eval_reward = self.evaluate_agent(
                    eval_env, eval_episodes=3
                )
                evaluation_list.append(
                    eval_reward.detach().cpu().item()
                )
        
        # Evaluate agent after training to assess new performance
        print("Training Complete.")
        eval_reward = self.evaluate_agent(
            eval_env, eval_episodes=10
        )
        evaluation_list.append(
            eval_reward.detach().cpu().item()
        )
        return WM_loss_list, actor_loss_list, critic_loss_list, evaluation_list

    def Run(self, env, env_seed, render=True):
        """Runs the agent for an episode on the environment.

        Args:
            env: Environment to roll out policy on, must respond to gymnasium commands.
            render (bool, optional): Choose whether to render the episode or not, gymnasium command.
                Defulats to True.
        """
        # Initialise total reward
        total_reward = 0

        # Reset env and manipulate observation and convert to a tensor
        observation, _ = env.reset(seed=env_seed)
        observation = observation.transpose(2,0,1).astype(np.uint8)
        obs_normalised = (observation.astype(np.float32) / 255.0) - 0.5
        observation_tensor = torch.tensor(
            obs_normalised, dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(0)

        # Initialise continue flag
        continue_ = True

        # Initialise hidden state and latent state with encoder
        hidden_state = torch.zeros(
            self.hidden_state_dims, dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            latent_state, _ = self.world_model.encoder.encode(
                hidden_state, observation_tensor
            )

        # Run until end of episode
        while continue_:
            # Render the episode if requested
            if render:
                env.render()

            # Generate action but cut gradient to ensure its not tracked
            with torch.no_grad():
                action, _, _ = self.agent.actor.act(
                    hidden_state, latent_state, deterministic=True
                )
            action_np = action.detach().cpu().numpy().squeeze(0).squeeze(0)

            # Step forward in env
            observation_, reward, terminated, truncated, _ = env.step(
                action_np
            )

            # Manipulate observation 
            observation_ = observation_.transpose(2,0,1).astype(np.uint8)
            obs__normalised = (observation_.astype(np.float32) / 255.0) - 0.5
            observation__tensor = torch.tensor(
                obs__normalised, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)

            # Log reward
            total_reward += reward

            # Check continue flags
            done = (terminated or truncated)
            continue_ = (1 - done)

            # Observe step
            with torch.no_grad():
                latent_state, hidden_state, _ = self.world_model.observe_step(
                    latent_state, hidden_state, action, observation__tensor
                )
            observation = observation_
            observation_tensor = observation__tensor
        return total_reward