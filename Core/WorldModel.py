import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import torch.distributions as distributions
from torch.distributions import Normal, Bernoulli
import torch.optim as optim

# file module imports
from Utils import to_twohot
from Networks import DynamicsPredictor, RewardPredictor, ContinuePredictor, SequenceModel, Decoder, Encoder

class WorldModel(nn.Module):
    """World Model class.

    This class is the World Model class, it consists of a Recurrant State Space Model (RSSM)
    and a Variational Auto Encoder (VAE). Training logic is self contained. This class
    allows the agent to imagine trajectories for training by learning the dynamics of the environment.

    Args:
        hidden_dims (int): Size of the hidden state vector.
        latent_dims (tuple(int, int)): Dimensions of the latent state matrix.
        observation_dims (tuple(int, int)): Dimensions of the observation.
        action_dims (int): Size of the action vector.
        training_horizon (int): Length of imagined trajectories.
        batch_size (int): Size of training batches.
        WM_lr (float): World model learn rate.
        WM_betas (tuple(float, float)): World Model adam optimiser hyperparams.
        WM_eps (float): World Model adam optimiser hyperparams.
        beta_pred (float): Relative loss scaling for prediction.
        beta_dyn (float): Relative loss sclaing for dynamics.
        beta_rep (float): Relative loss scaling for representation.
        num_encoder_filters_1 (int): Number of filters in the encoder layer 1.
        num_encoder_filters_2 (int): Number of filters in the encoder layer 2.
        encoder_hidden_layer_nodes (int): Number of neurons in the encoder HL.
        num_decoder_filters_1 (int): Number of filters in the decoder layer 1.
        num_decoder_filters_2 (int): Number of filters in the decoder layer 2.
        decoder_hidden_layer_nodes (int): Number of neurons in the decoder HL.
        dyn_pred_hidden_num_nodes_1 (int): Number of neurons in the dyn predictor network HL1.
        dyn_pred_hidden_num_nodes_2 (int): Number of neurons in the dyn predictor network HL2.
        rew_pred_hidden_num_nodes_1 (int): Number of neurons in the rew predictor network HL1.
        rew_pred_hidden_num_nodes_2 (int): Number of neurons in the rew predictor network HL2.
        reward_buckets (int): Number of reward buckets in twohot encoding.
        cont_pred_hidden_num_nodes_1 (int): Number of neurons in the cont predictor network HL1.
        cont_pred_hidden_num_nodes_2 (int): Number of neurons in the cont predictor network HL2.
        device (str, optional): Storage location of the network ('cpu' or 'cuda'). 
            Defaults to 'cpu'.

    Attributes:
        latent_num_rows (int): Number of rows in latent state matrix.
        latent_num_columns (int): Number of columns in latent state matrix.
        hidden_dims (int): Size of hidden state vector.
        action_dims (int): Size of action vector.
        observation_dim_x (int): Number of columns in observation.
        observation_dim_y (int): Number of rows in observation.
        horizon (int): Length of imagination horizon.
        buckets (int): Number of buckets in twohot encoding.
        device (str): Storage location of the network ('cpu' or 'cuda').
    """
    def __init__(
            self, hidden_dims: int, latent_dims: int,
            observation_dims: int, action_dims: int, training_horizon: int,
            batch_size: int, WM_lr: float, WM_betas: tuple[float, float],
            WM_eps: float, beta_pred: float, beta_dyn: float,
            beta_rep, num_encoder_filters_1: int, num_encoder_filters_2: int,
            encoder_hidden_layer_nodes: int, num_decoder_filters_1: int, num_decoder_filters_2: int,
            decoder_hidden_layer_nodes: int, dyn_pred_hidden_num_nodes_1: int, dyn_pred_hidden_num_nodes_2: int,
            rew_pred_hidden_num_nodes_1: int, rew_pred_hidden_num_nodes_2: int, reward_buckets: int,
            cont_pred_hidden_num_nodes_1: int, cont_pred_hidden_num_nodes_2: int,
            device='cpu'
        ):
        super().__init__()
        self.latent_num_rows, self.latent_num_columns = latent_dims
        self.hidden_dims = hidden_dims
        self.action_dims = action_dims
        self.observation_dim_x, self.observation_dim_y = observation_dims
        self.horizon = training_horizon
        self.buckets = reward_buckets

        self.beta_pred = beta_pred
        self.beta_dyn = beta_dyn
        self.beta_rep = beta_rep
        self.batch_size = batch_size

        self.encoder = Encoder(
            observation_dims, hidden_dims, latent_dims[0], 
            latent_dims[1], num_encoder_filters_1, num_encoder_filters_2, 
            encoder_hidden_layer_nodes, device=device
        )
        self.sequence_model = SequenceModel(
            latent_dims[0], latent_dims[1], hidden_dims, 
            action_dims, num_layers=1, device=device
        )
        self.dynamics_predictor = DynamicsPredictor(
            latent_dims[0], latent_dims[1], hidden_dims,
            dyn_pred_hidden_num_nodes_1, dyn_pred_hidden_num_nodes_2, 
            device=device
        )
        self.reward_predictor = RewardPredictor(
            latent_dims[0], latent_dims[1], hidden_dims,
            rew_pred_hidden_num_nodes_1, rew_pred_hidden_num_nodes_2,
            reward_buckets, device=device
        )
        self.continue_predictor = ContinuePredictor(
            latent_dims[0], latent_dims[1], hidden_dims,
            cont_pred_hidden_num_nodes_1, cont_pred_hidden_num_nodes_2,
            device=device
        )
        self.decoder = Decoder(
            latent_dims[0], latent_dims[1], observation_dims,
            hidden_dims, num_decoder_filters_1, num_decoder_filters_2,
            decoder_hidden_layer_nodes, device=device
        )
        self.device = device
 
        self.optimiser = torch.optim.AdamW(
            self.parameters(),
            lr=WM_lr,
            betas=(WM_betas[0], WM_betas[1]),
            eps=WM_eps,
            weight_decay=1e-6
        )

        # Gradscalar to accelarate training
        self.scalar = torch.amp.GradScaler()

    def imagine_step(
            self, hidden_state: torch.Tensor, 
            latent_state: torch.Tensor, action: torch.Tensor
        ):
        """Dreams steps forward into the future taking a given action.
        
        This function uses the RSSM to take an action from a given
        state and imagine the future response.

        Args:
            hidden_state (torch.Tensor): Hidden state to dream from.
            latent_state (torch.Tensor): latent state to dream from.
            action (torch.Tensor): action to dream the response to.

        Returns:
            next_hidden_state (torch.Tensor): Imagined next hidden state.
            next_latent_state (torch.Tensor): Imagined next latent state.
            next_reward (torch.Tensor): Imagined next reward.
            continue_ (torch.Tensor): Imagined continue flag.
        """
        # Predict response with RSSM
        next_hidden_state = self.sequence_model(
            latent_state, hidden_state, action
        )
        next_latent_state, _ = self.dynamics_predictor.predict(
            next_hidden_state
        )
        next_reward = self.reward_predictor.predict(
            next_hidden_state, next_latent_state
        )
        continue_ = self.continue_predictor.predict(
            next_hidden_state, next_latent_state
        )
        return next_hidden_state, next_latent_state, next_reward, continue_

    def observe_step(
            self, last_latent: torch.Tensor, last_hidden: torch.Tensor,
            last_action: torch.Tensor, observation: torch.Tensor
        ):
        """Processes observed data from the environment into the RSSM.
        
        Data from a step in the environment must be passed through the RSSM
        to process it so that the hidden and latent states can be found 
        for the next step.

        Args:
            last_hidden (torch.Tensor): Hidden state action was taken in.
            last_latent (torch.Tensor): Latent state action was taken in.
            last_action (torch.Tensor): Action taken in environment.
            observation (torch.Tensor): action to dream the response to.

        Returns:
            latent_state (torch.Tensor): Return latent state.
            hidden_state (torch.Tensor): Return hidden state.
            latent_logits (torch.Tensor): Logits for latent state prediction.
        """
        # Encodes response into RSSM to find new returns
        hidden_state = self.sequence_model.forward(
            last_latent, last_hidden, last_action
        )
        latent_state, latent_logits = self.encoder.encode(
            hidden_state, observation
        )
        return latent_state, hidden_state, latent_logits

    def unroll_model(
            self, observation_sequence_batch: torch.Tensor,
            action_sequence_batch: torch.Tensor, reward_sequence_batch: torch.Tensor,
            continue_sequence_batch: torch.Tensor
        ):
        """Unrolls model to compute priors and posteriors for training.
        
        In order to train the world model, .

        Args:
            observation_sequence_batch (torch.Tensor): Sampled batch of observation sequences.
            action_sequence_batch (torch.Tensor): Sampled batch of action sequences.
            reward_sequence_batch (torch.Tensor): Sampled batch of rewards.
            continue_sequence_batch (torch.Tensor): Sampled batch of continue flag sequences.

        Returns:
            prior_logits (torch.Tensor): Prediction logits for the prior.
            posterior_logits (torch.Tensor): Prediction logits for the posterior.
            obs_log_lh (torch.Tensor): Observation prediction log likelyhood.
            rew_log_lh (torch.Tensor): Reward prediction log likelyhood.
            cont_log_lh (torch.Tensor): Continue prediction log likelyhood.
        """
        # Initialise hidden and latent state
        B = continue_sequence_batch.shape[0]
        hidden_state_batch = torch.zeros(
            B, 1, self.hidden_dims, dtype=torch.float32, device=self.device
        )
        posterior_latent_batch = torch.zeros(
            B, 1, self.latent_num_rows, self.latent_num_columns,
            dtype=torch.float32, device=self.device
        )
        post_latent_list, hidden_list, logits_list = [], [], []

        # Loop through sequence
        for t in range(self.horizon):
            # a_t is ahead of t in time so if t=0 then set a to zero vector
            if t > 0:
                prev_action_batch = action_sequence_batch[:, t-1:t]
            else:
                prev_action_batch = torch.zeros(
                    B, 1, self.action_dims, device=self.device
                )

            # Compute posterior
            posterior_latent_batch, hidden_state_batch, posterior_logits_batch = self.observe_step(
                posterior_latent_batch, hidden_state_batch,
                prev_action_batch, observation_sequence_batch[:, t:t+1]
            )
            post_latent_list.append(posterior_latent_batch)
            hidden_list.append(hidden_state_batch)
            logits_list.append(posterior_logits_batch)

        # Concatenate list in sequence dimension
        posterior_logits = torch.cat(logits_list, dim=1) # (B, H, 32, 32)
        hidden_seq = torch.cat(hidden_list, dim=1)          # (B, H, Hidden)
        latent_seq = torch.cat(post_latent_list, dim=1)      # (B, H, 32, 32)

        # Shift forward in time to remove zero intialisation
        pred_hidden_input = hidden_seq[:, 1:]
        pred_latent_input = latent_seq[:, 1:]

        # Compute priors using RSSM
        prior_logits = self.dynamics_predictor(hidden_seq)
        dec_mu = self.decoder(hidden_seq, latent_seq)
        reward_logits = self.reward_predictor(pred_hidden_input, pred_latent_input)
        _, cont_logits = self.continue_predictor(pred_hidden_input, pred_latent_input)

        # Slice buffer data to match time horizon
        obs_targets = observation_sequence_batch[:, :self.horizon]
        rew_targets = reward_sequence_batch[:, :self.horizon - 1]
        cont_targets = continue_sequence_batch[:, :self.horizon - 1]

        # Convert reward targets to twohot encodings
        reward_th = to_twohot(rew_targets, self.reward_predictor.buckets_rew)

        # Compute log likelyhood of the reconstructed observation, this can be found 
        # using the MSE between the mean decoder output and the target reconstructed
        obs_log_lh = - (dec_mu.float() - obs_targets.float()).pow(2).sum(dim=[-3,-2,-1])

        # Compute continue prediction log likelyhood with binary cross entropy
        cont_log_lh = torch.nn.functional.binary_cross_entropy_with_logits(
            cont_logits, cont_targets, reduction='none'
        )

        # Compute reward log likelyhood with categorical cross entropy
        reward_log_probs = torch.nn.functional.log_softmax(
            reward_logits, dim=-1
        )
        rew_log_lh = torch.sum(
            reward_th * reward_log_probs, dim=-1, keepdim=True
        )

        # Slice the logits and log likelyhood to align with the time
        # shifted reward and continue targets
        return (
            prior_logits[:, 1:], posterior_logits[:, 1:], obs_log_lh[:, 1:],
            rew_log_lh, cont_log_lh
        )

    def training_step(
            self, observation_sequences: torch.Tensor, action_sequences: torch.Tensor,
            reward_sequences: torch.Tensor, continue_sequences: torch.Tensor
        ): # (batch_size, sequence_length, features)
        """Runs a training step for the world model with data sampled from the buffer.
        
        Trains the world model using data sampled from the buffer and returns loss from step.

        Args:
            observation_sequences (torch.Tensor): Sampled batch of observation sequences.
            action_sequences (torch.Tensor): Sampled batch of action sequences.
            reward_sequences (torch.Tensor): Sampled batch of rewards.
            continue_sequences (torch.Tensor): Sampled batch of continue flag sequences.

        Returns:
            total_loss (torch.Tensor): Loss from training step to log during training.
        """
        # Normalise image and slice samples to match horizon
        observation_sequences = (observation_sequences.float() / 255.0) - 0.5
        obs_slice = observation_sequences[:, :self.horizon]
        a_slice = action_sequences[:, :self.horizon]
        r_slice = reward_sequences[:, :self.horizon]
        c_slice = continue_sequences[:, :self.horizon]

        # Wrap training in AMP to accelarate FP operations
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            # Compute posterior and priors
            (prior_logits, posterior_logits, obs_log_lh, 
             rew_log_lh, cont_log_lh) = self.unroll_model(
                obs_slice, a_slice, r_slice, c_slice
            )

            # Mask all dead states to zero gradients for dead states as the 
            # agent shouldnt be able to predict post death scene
            mask = continue_sequences[:, :self.horizon - 1]
            obs_log_lh = obs_log_lh * mask.squeeze(-1)
            rew_log_lh = rew_log_lh * mask
            cont_log_lh = cont_log_lh * mask

            # Create prior and posterior distributions, one attached 
            # and one detached of both
            prior_dist_detached = torch.distributions.Categorical(
                logits=prior_logits.detach().float())
            posterior_dist_detached = torch.distributions.Categorical(
                logits=posterior_logits.detach().float())
            prior_dist = torch.distributions.Categorical(
                logits=prior_logits.float())
            posterior_dist = torch.distributions.Categorical(
                logits=posterior_logits.float())

            # Dynamics loss: KL divergence to force the prior to match the posterior
            Dkl_dyn = torch.distributions.kl.kl_divergence(
                posterior_dist_detached, prior_dist).sum(dim=-1)
            Dkl_dyn = torch.mean(
                Dkl_dyn * mask.squeeze(-1))
            loss_dyn = torch.max(torch.tensor(1.0, device=self.device), Dkl_dyn)

            # Representation loss: KL divergence to force the posterior network to
            # build representations that the prior network can easily predict from
            Dkl_rep = torch.distributions.kl.kl_divergence(
                posterior_dist, prior_dist_detached).sum(dim=-1)
            Dkl_rep = torch.mean(
                Dkl_rep * mask.squeeze(-1))
            loss_rep = torch.max(torch.tensor(1.0, device=self.device), Dkl_rep)

            # Prediction loss: accounts for 3 independent outputs of the RSSM and prevent
            # representation collapse by forcing a useful output and therefore a useful representation
            denominator = mask.sum() + 1e-5
            loss_pred = (- obs_log_lh.sum() - rew_log_lh.sum() + cont_log_lh.sum()) / denominator

            # Total loss weighted by hyperparams
            total_loss = self.beta_pred * loss_pred + self.beta_dyn * loss_dyn + self.beta_rep * loss_rep

            # Check for dangerous losses
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("World Model loss is nan or inf, skipping update.")
                return total_loss

        # Perform backprop update with AMP scaled gradients
        self.optimiser.zero_grad()
        self.scalar.scale(total_loss).backward()
        self.scalar.unscale_(self.optimiser)
        nn.utils.clip_grad_norm_(self.parameters(), 100.0)
        self.scalar.step(self.optimiser)
        self.scalar.update()

        return total_loss