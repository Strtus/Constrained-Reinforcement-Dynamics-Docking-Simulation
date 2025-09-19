"""
Advanced Safe Reinforcement Learning Agent for Spacecraft Docking
================================================================

Implementation of Guided Reward Policy Optimization (GRPO)
with safety constraints for autonomous spacecraft rendezvous and docking.
Incorporates Lagrangian constraint optimization and advanced neural architectures.

Author: Strtus
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, namedtuple
import logging
from dataclasses import dataclass
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done', 
    'constraint_costs', 'lagrange_multipliers', 'advantage'
])


@dataclass
class GRPOConfig:
    """Configuration for GRPO algorithm"""
    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 3
    attention_heads: int = 8
    transformer_layers: int = 2
    
    # Training parameters
    learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    
    # GRPO specific
    trust_region_coeff: float = 0.1
    constraint_violation_penalty: float = 10.0
    lagrange_lr: float = 0.01
    adaptive_constraint_threshold: float = 0.1
    
    # Safety parameters
    safe_exploration_std: float = 0.1
    constraint_tolerance: float = 1e-3
    safety_critic_weight: float = 1.0
    
    # Training schedule
    update_frequency: int = 4
    target_update_frequency: int = 100
    evaluation_frequency: int = 1000


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for state encoding"""
    
    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert input_dim % num_heads == 0
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.query_projection = nn.Linear(input_dim, input_dim)
        self.key_projection = nn.Linear(input_dim, input_dim)
        self.value_projection = nn.Linear(input_dim, input_dim)
        self.output_projection = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Generate queries, keys, values
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)
        
        # Concatenate heads and project
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.input_dim
        )
        output = self.output_projection(attended_values)
        
        # Residual connection and layer normalization
        return self.layer_norm(x + output)


class StateEncoder(nn.Module):
    """Advanced state encoder with attention mechanism"""
    
    def __init__(self, input_dim: int, hidden_dim: int, attention_heads: int = 8):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, attention_heads)
        
        # Physical feature extractors
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.velocity_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.attitude_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),  # Quaternion
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.system_encoder = nn.Sequential(
            nn.Linear(13, hidden_dim // 4),  # Angular velocity + thruster health + fuel
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state with physical feature extraction
        
        Args:
            state: [batch, 26] - [pos(3), vel(3), quat(4), omega(3), thruster(12), fuel(1)]
        """
        batch_size = state.shape[0]
        
        # Extract physical components
        position = state[:, 0:3]
        velocity = state[:, 3:6]
        attitude = state[:, 6:10]
        system_state = state[:, 10:]  # angular velocity + thruster health + fuel
        
        # Encode each component
        pos_features = self.position_encoder(position)
        vel_features = self.velocity_encoder(velocity)
        att_features = self.attitude_encoder(attitude)
        sys_features = self.system_encoder(system_state)
        
        # Concatenate features
        combined_features = torch.cat([pos_features, vel_features, att_features, sys_features], dim=1)
        
        # Apply attention mechanism (treat as sequence of 1)
        features_seq = combined_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        attended_features = self.attention(features_seq).squeeze(1)  # [batch, hidden_dim]
        
        # Final feature fusion
        encoded_state = self.feature_fusion(attended_features)
        
        return encoded_state


class SafetyAwareActorNetwork(nn.Module):
    """Safety-aware policy network with constraint consideration"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, 
                 attention_heads: int = 8, num_layers: int = 3):
        super().__init__()
        
        self.action_dim = action_dim
        
        # State encoding
        self.state_encoder = StateEncoder(state_dim, hidden_dim, attention_heads)
        
        # Policy network layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1)
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1)
                ])
        
        self.policy_layers = nn.Sequential(*layers)
        
        # Action mean and std networks
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_std = nn.Linear(hidden_dim, action_dim)
        
        # Safety-aware action scaling
        self.safety_scaler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Sigmoid()  # Scale factor [0, 1]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, 
                constraint_violations: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with safety-aware action generation
        
        Args:
            state: State tensor [batch, state_dim]
            constraint_violations: Constraint violation indicators [batch, num_constraints]
            
        Returns:
            action_mean: Mean of action distribution [batch, action_dim]
            action_std: Standard deviation of action distribution [batch, action_dim]
        """
        # Encode state
        encoded_state = self.state_encoder(state)
        
        # Process through policy layers
        policy_features = self.policy_layers(encoded_state)
        
        # Generate action parameters
        action_mean = self.action_mean(policy_features)
        action_log_std = self.action_std(policy_features)
        
        # Apply safety scaling based on constraint violations
        if constraint_violations is not None:
            safety_scale = self.safety_scaler(encoded_state)
            # Reduce action magnitude when constraints are violated
            constraint_penalty = torch.sum(constraint_violations, dim=1, keepdim=True)
            safety_factor = torch.exp(-constraint_penalty.unsqueeze(-1))
            action_mean = action_mean * safety_scale * safety_factor
        
        # Ensure std is positive and bounded
        action_std = torch.exp(torch.clamp(action_log_std, -2, 2))
        
        return action_mean, action_std
    
    def sample_action(self, state: torch.Tensor, 
                     constraint_violations: Optional[torch.Tensor] = None,
                     deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy
        
        Returns:
            action: Sampled action [batch, action_dim]
            log_prob: Log probability of action [batch, 1]
        """
        action_mean, action_std = self.forward(state, constraint_violations)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros(action_mean.shape[0], 1, device=action_mean.device)
        else:
            # Create multivariate normal distribution
            action_dist = Normal(action_mean, action_std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=1, keepdim=True)
        
        # Clip action to valid range
        action = torch.tanh(action)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """Value function critic with safety-aware value estimation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        
        # State encoding
        self.state_encoder = StateEncoder(state_dim, hidden_dim)
        
        # Value network
        layers = []
        input_dim = hidden_dim + action_dim
        
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1)
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1)
                ])
        
        self.value_layers = nn.Sequential(*layers)
        
        # Output layers
        self.value_head = nn.Linear(hidden_dim, 1)
        self.safety_value_head = nn.Linear(hidden_dim, 4)  # One per constraint type
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            state: State tensor [batch, state_dim]
            action: Action tensor [batch, action_dim]
            
        Returns:
            value: Value estimate [batch, 1]
            safety_values: Safety constraint value estimates [batch, 4]
        """
        # Encode state
        encoded_state = self.state_encoder(state)
        
        # Concatenate state and action
        state_action = torch.cat([encoded_state, action], dim=1)
        
        # Process through value layers
        value_features = self.value_layers(state_action)
        
        # Generate outputs
        value = self.value_head(value_features)
        safety_values = self.safety_value_head(value_features)
        
        return value, safety_values


class GRPOAgent:
    """Guided Reward Policy Optimization Agent with Safety Constraints"""
    
    def __init__(self, state_dim: int, action_dim: int, config: GRPOConfig = None):
        """Initialize GRPO agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Algorithm configuration
        """
        self.config = config or GRPOConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize networks
        self.actor = SafetyAwareActorNetwork(
            state_dim, action_dim, self.config.hidden_dim,
            self.config.attention_heads, self.config.num_layers
        ).to(self.device)
        
        self.critic = CriticNetwork(
            state_dim, action_dim, self.config.hidden_dim, self.config.num_layers
        ).to(self.device)
        
        self.target_critic = CriticNetwork(
            state_dim, action_dim, self.config.hidden_dim, self.config.num_layers
        ).to(self.device)
        
        # Copy parameters to target network
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(), lr=self.config.learning_rate, weight_decay=1e-4
        )
        
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(), lr=self.config.critic_learning_rate, weight_decay=1e-4
        )
        
        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=100000, eta_min=1e-6
        )
        
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=100000, eta_min=1e-6
        )
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=self.config.buffer_size)
        
        # Lagrangian multipliers for constraints (learnable parameters)
        self.lagrange_multipliers = torch.zeros(4, device=self.device, requires_grad=True)
        self.lagrange_optimizer = optim.Adam([self.lagrange_multipliers], lr=self.config.lagrange_lr)
        
        # Training statistics
        self.training_stats = {
            'actor_loss': deque(maxlen=1000),
            'critic_loss': deque(maxlen=1000),
            'constraint_violations': deque(maxlen=1000),
            'safety_value_loss': deque(maxlen=1000),
            'entropy': deque(maxlen=1000)
        }
        
        # Episode tracking
        self.episode_count = 0
        self.update_count = 0
        
    def select_action(self, state: np.ndarray, 
                     constraint_violations: Optional[np.ndarray] = None,
                     deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using current policy
        
        Args:
            state: Current state
            constraint_violations: Current constraint violations
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Selected action
            info: Additional information (log_prob, etc.)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            constraint_tensor = None
            if constraint_violations is not None:
                constraint_tensor = torch.FloatTensor(constraint_violations).unsqueeze(0).to(self.device)
            
            action, log_prob = self.actor.sample_action(
                state_tensor, constraint_tensor, deterministic
            )
            
            action_np = action.cpu().numpy().flatten()
            log_prob_np = log_prob.cpu().numpy().flatten()
            
        return action_np, {'log_prob': log_prob_np}
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool, constraint_costs: np.ndarray,
                        lagrange_multipliers: np.ndarray):
        """Store experience in replay buffer"""
        # Calculate advantage (will be computed during training)
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            constraint_costs=constraint_costs,
            lagrange_multipliers=lagrange_multipliers,
            advantage=0.0  # Will be computed later
        )
        
        self.replay_buffer.append(experience)
    
    def compute_gae_advantages(self, experiences: List[Experience]) -> List[float]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        # Convert experiences to tensors
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.FloatTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        
        with torch.no_grad():
            values, _ = self.critic(states, actions)
            values = values.squeeze()
            
            next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
            next_actions, _ = self.actor.sample_action(next_states, deterministic=True)
            next_values, _ = self.critic(next_states, next_actions)
            next_values = next_values.squeeze()
            
        # Compute advantages using GAE
        for i in reversed(range(len(experiences))):
            if dones[i]:
                next_value = 0
            else:
                next_value = next_values[i]
            
            delta = rewards[i] + self.config.gamma * next_value - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages.insert(0, gae.item())
            
            if dones[i]:
                gae = 0
        
        return advantages
    
    def update_policy(self) -> Dict[str, float]:
        """Update policy using GRPO algorithm"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        # Sample batch from replay buffer
        batch_experiences = list(self.replay_buffer)[-self.config.batch_size:]
        
        # Compute advantages
        advantages = self.compute_gae_advantages(batch_experiences)
        
        # Update experiences with advantages
        for i, exp in enumerate(batch_experiences):
            batch_experiences[i] = exp._replace(advantage=advantages[i])
        
        # Convert to tensors
        states = torch.FloatTensor([exp.state for exp in batch_experiences]).to(self.device)
        actions = torch.FloatTensor([exp.action for exp in batch_experiences]).to(self.device)
        old_log_probs = torch.FloatTensor([0.0 for _ in batch_experiences]).to(self.device)  # Placeholder
        rewards = torch.FloatTensor([exp.reward for exp in batch_experiences]).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        constraint_costs = torch.FloatTensor([exp.constraint_costs for exp in batch_experiences]).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Actor update
        self.actor_optimizer.zero_grad()
        
        action_mean, action_std = self.actor(states)
        action_dist = Normal(action_mean, action_std)
        new_log_probs = action_dist.log_prob(actions).sum(dim=1)
        entropy = action_dist.entropy().sum(dim=1)
        
        # PPO-style ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages_tensor
        
        # Policy loss with safety constraints
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -self.config.entropy_coeff * entropy.mean()
        
        # Constraint penalty using Lagrangian multipliers
        constraint_penalty = torch.sum(self.lagrange_multipliers.unsqueeze(0) * constraint_costs, dim=1).mean()
        
        actor_loss = policy_loss + entropy_loss + constraint_penalty
        
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()
        
        # Critic update
        self.critic_optimizer.zero_grad()
        
        values, safety_values = self.critic(states, actions)
        value_targets = rewards + self.config.gamma * advantages_tensor  # Simplified target
        
        value_loss = F.mse_loss(values.squeeze(), value_targets)
        
        # Safety value loss
        safety_value_loss = F.mse_loss(safety_values, constraint_costs)
        
        critic_loss = value_loss + self.config.safety_critic_weight * safety_value_loss
        
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()
        
        # Update Lagrangian multipliers
        self.lagrange_optimizer.zero_grad()
        constraint_violation = torch.sum(constraint_costs, dim=1).mean()
        lagrange_loss = -torch.sum(self.lagrange_multipliers) * (constraint_violation - self.config.adaptive_constraint_threshold)
        lagrange_loss.backward()
        self.lagrange_optimizer.step()
        
        # Ensure multipliers are non-negative
        with torch.no_grad():
            self.lagrange_multipliers.clamp_(min=0)
        
        # Update target network
        if self.update_count % self.config.target_update_frequency == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Update learning rate
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # Store training statistics
        self.training_stats['actor_loss'].append(actor_loss.item())
        self.training_stats['critic_loss'].append(critic_loss.item())
        self.training_stats['constraint_violations'].append(constraint_violation.item())
        self.training_stats['safety_value_loss'].append(safety_value_loss.item())
        self.training_stats['entropy'].append(entropy.mean().item())
        
        self.update_count += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'constraint_violations': constraint_violation.item(),
            'safety_value_loss': safety_value_loss.item(),
            'entropy': entropy.mean().item(),
            'lagrange_multipliers': self.lagrange_multipliers.detach().cpu().numpy()
        }
    
    def save_model(self, path: str):
        """Save model parameters"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'lagrange_multipliers': self.lagrange_multipliers,
            'config': self.config,
            'training_stats': dict(self.training_stats),
            'episode_count': self.episode_count,
            'update_count': self.update_count
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model parameters"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.lagrange_multipliers = checkpoint['lagrange_multipliers']
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        if 'episode_count' in checkpoint:
            self.episode_count = checkpoint['episode_count']
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']
        
        logger.info(f"Model loaded from {path}")
    
    def get_training_statistics(self) -> Dict[str, float]:
        """Get current training statistics"""
        stats = {}
        
        for key, values in self.training_stats.items():
            if len(values) > 0:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_latest'] = values[-1]
        
        stats['episode_count'] = self.episode_count
        stats['update_count'] = self.update_count
        stats['replay_buffer_size'] = len(self.replay_buffer)
        
        return stats