from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Sequence


class DSRLAgent(nn.Module):
    """DSRL agent - https://arxiv.org/abs/2506.15799"""

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_flow_actor,
        make_bc_flow_actor_optimizer,
        make_noise_actor,
        make_noise_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_z_critic,
        make_z_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        noise_scale: float = 1.0,

        online_training: bool = False,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.noise_scale = noise_scale
        self.target_entropy = -action_dim

        self.bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor.load_state_dict(self.bc_flow_actor.state_dict())

        self.noise_actor = make_noise_actor(observation_shape, action_dim)

        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.z_critic = make_z_critic(observation_shape, action_dim)

        self.log_alpha = nn.Parameter(torch.zeros((), device=ptu.device))

        self.bc_flow_actor_optimizer = make_bc_flow_actor_optimizer(self.bc_flow_actor.parameters())
        self.noise_actor_optimizer = make_noise_actor_optimizer(self.noise_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.z_critic_optimizer = make_z_critic_optimizer(self.z_critic.parameters())
        self.alpha_optimizer = make_noise_actor_optimizer([self.log_alpha])

        self.to(ptu.device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.compiler.disable
    def sample_flow_actions(self, observations: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """Euler integration of BC flow from t=0 to t=1."""
        h = 1.0 / self.flow_steps
        action = noises + 0
        for k in range(self.flow_steps):
            action = action + h * self.target_bc_flow_actor(observations, action, torch.full((*noises.shape[:-1], 1), k * h, device=noises.device))
        return torch.clamp(action, -1, 1)

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample actions using noise policy for noise input to BC flow policy."""
        return self.sample_flow_actions(observations, self.noise_scale * self.noise_actor(observations).rsample())
    
    def get_action(self, observation: np.ndarray):
        """Used for evaluation."""
        return ptu.to_numpy(self.sample_actions(ptu.from_numpy(np.asarray(observation))[None]))[0]

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """Update critic"""
        with torch.no_grad():
            z = self.noise_actor(next_observations).rsample()
            a = self.sample_flow_actions(next_observations, self.noise_scale * z)
            y = rewards + self.discount * (1 - dones) * self.target_critic(next_observations, a).min(0)[0]

        q = self.critic(observations, actions)
        loss = ((q - y[None]) ** 2).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }
    
    def update_qz(self, 
        observations: torch.Tensor,
        actions: torch.Tensor,
        noises: torch.Tensor,
    ) -> dict:
        """Update z_critic."""

        with torch.no_grad():
            a = self.sample_flow_actions(observations, self.noise_scale * noises)
            y = self.critic(observations, a).min(0)[0]

        q = self.z_critic(observations, self.noise_scale * noises)
        loss = ((q - y[None]) ** 2).mean()

        self.z_critic_optimizer.zero_grad()
        loss.backward()
        self.z_critic_optimizer.step()

        return {
            "z_q_loss": loss,
            "z_q_mean": q.mean(),
        }

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """Update BC flow actor"""
        z = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], 1, device=actions.device)
        loss = ((self.bc_flow_actor(observations, (1 - t) * z + t * actions, t) - (actions - z)) ** 2).mean()

        self.bc_flow_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_flow_actor_optimizer.step()

        return {
            "actor_loss": loss,
        }
    
    def update_noise_actor(self,
        observations: torch.Tensor,
    ) -> dict:
        """Update noise actor."""
        p = self.noise_actor(observations)
        z = p.rsample()
        loss = (self.alpha.detach() * p.log_prob(z) - self.z_critic(observations, self.noise_scale * z).min(0)[0]).mean()

        self.noise_actor_optimizer.zero_grad()
        loss.backward()
        self.noise_actor_optimizer.step()

        return {
            "noise_actor_loss": loss,
        }

    def update_alpha(self, observations: torch.Tensor) -> dict:
        """Update alpha."""
        p = self.noise_actor(observations)
        loss = -(self.alpha * (p.log_prob(p.rsample()) + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()

        return {
            "alpha_loss": loss,
            "alpha": self.alpha,
        }

    def update_target_critic(self) -> None:
        for p, q in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            p.data.copy_(self.target_update_rate * q.data + (1 - self.target_update_rate) * p.data)

    def update_target_bc_flow_actor(self) -> None:
        for p, q in zip(self.target_bc_flow_actor.parameters(), self.bc_flow_actor.parameters()):
            p.data.copy_(self.target_update_rate * q.data + (1 - self.target_update_rate) * p.data)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        z = torch.randn_like(actions)
        mq = self.update_q(observations, actions, rewards, next_observations, dones)
        mz = self.update_qz(observations, actions, z)
        ma = self.update_actor(observations, actions)
        mn = self.update_noise_actor(observations)
        malpha = self.update_alpha(observations)
        metrics = {
            "q_loss": mq["q_loss"].item(),
            "q_mean": mq["q_mean"].item(),
            "q_max": mq["q_max"].item(),
            "q_min": mq["q_min"].item(),
            "z_q_loss": mz["z_q_loss"].item(),
            "z_q_mean": mz["z_q_mean"].item(),
            "actor_loss": ma["actor_loss"].item(),
            "noise_actor_loss": mn["noise_actor_loss"].item(),
            "alpha_loss": malpha["alpha_loss"].item(),
            "alpha": malpha["alpha"].item(),
        }

        self.update_target_critic()
        self.update_target_bc_flow_actor()

        return metrics

