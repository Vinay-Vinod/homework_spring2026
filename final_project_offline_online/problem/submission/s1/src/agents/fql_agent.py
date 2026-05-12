from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        z = torch.randn(1, self.action_dim, device=observation.device)
        action = self.onestep_actor(observation, z)
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        action = noise + 0
        step_size = 1.0 / self.flow_steps
        for step_idx in range(self.flow_steps):
            time_frac = torch.full(
                (*noise.shape[:-1], 1), step_idx * step_size, device=noise.device
            )
            action = action + step_size * self.bc_actor(observation, action, time_frac)
        action = torch.clamp(action, -1, 1)
        return action

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        with torch.no_grad():
            z_next = torch.randn_like(actions)
            a_next = self.onestep_actor(next_observations, z_next)
            a_next = torch.clamp(a_next, -1, 1)
            q_target_pair = self.target_critic(next_observations, a_next)
            q_next_mean = q_target_pair.mean(dim=0)
            td_target = rewards + self.discount * (1 - dones) * q_next_mean

        q = self.critic(observations, actions)
        loss = ((q - td_target[None]) ** 2).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        z = torch.randn_like(actions)
        time_frac = torch.rand(actions.shape[0], 1, device=actions.device)
        a_interp = (1 - time_frac) * z + time_frac * actions
        flow_target = actions - z

        v_pred = self.bc_actor(observations, a_interp, time_frac)
        loss = ((v_pred - flow_target) ** 2).mean(dim=-1).mean()

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        z = torch.randn_like(actions)
        a_onestep = self.onestep_actor(observations, z)

        a_bc = self.get_bc_action(observations, z).detach()

        distill_loss = self.alpha * ((a_onestep - a_bc) ** 2).mean(dim=-1).mean()

        a_onestep_clipped = torch.clamp(a_onestep, -1, 1)
        q_onestep = self.critic(observations, a_onestep_clipped)
        q_loss = -q_onestep.mean(dim=0).mean()

        loss = distill_loss + q_loss

        with torch.no_grad():
            mse = ((a_onestep - a_bc) ** 2).mean(dim=-1).mean()

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        for target_param, online_param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.target_update_rate * online_param.data
                + (1 - self.target_update_rate) * target_param.data
            )