from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class IFQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor_flow,
        make_actor_flow_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        num_samples: int = 32,
        expectile: float = 0.9,
        **kwargs,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.actor_flow = make_actor_flow(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = make_value(observation_shape)

        self.actor_flow_optimizer = make_actor_flow_optimizer(self.actor_flow.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.num_samples = num_samples
        self.expectile = expectile
        self.to(ptu.device)

    @staticmethod
    def expectile_loss(adv: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Compute the expectile loss for IFQL
        """
        return torch.where(adv > 0, torch.full_like(adv, expectile), torch.full_like(adv, 1 - expectile)) * adv ** 2

    @torch.compile
    def update_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """
        Update value function
        """
        with torch.no_grad():
            q = self.target_critic(observations, actions).min(dim=0).values
        v = self.value(observations)
        loss = self.expectile_loss(q - v, self.expectile).mean()

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        return {
            "loss": loss,
            "v_mean": v.mean(),
            "adv_mean": (q - v).mean(),
        }

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Rejection / best-of-n sampling using the flow policy and critic.

        We:
          1. Sample multiple candidate actions via the BC flow.
          2. Evaluate them with the critic.
          3. Pick the action with the highest Q-value.
        """
        b = observations.shape[0]
        obs = observations.repeat_interleave(self.num_samples, dim=0)
        candidate = self.get_flow_action(obs, torch.randn(obs.shape[0], self.action_dim, device=obs.device))
        q = self.critic(obs, candidate).min(0)[0].view(b, self.num_samples)
        return candidate.view(b, self.num_samples, self.action_dim)[torch.arange(b, device=candidate.device), q.argmax(1)]

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        return ptu.to_numpy(self.sample_actions(ptu.from_numpy(np.asarray(observation))[None]))[0]

    @torch.compile
    def get_flow_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Compute the flow action using Euler integration for `self.flow_steps` steps.
        """
        h = 1.0 / self.flow_steps
        action = noise + 0
        for k in range(self.flow_steps):
            action = action + h * self.actor_flow(observation, action, torch.full((*noise.shape[:-1], 1), k * h, device=noise.device))
        return torch.clamp(action, -1, 1)

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
        Update Q(s, a) using the learned value function for bootstrapping,
        as in IFQL / IQL-style critic training.
        """
        with torch.no_grad():
            y = rewards + self.discount * (1 - dones) * self.value(next_observations)
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


    @torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the flow actor using the velocity matching loss.
        """
        z = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], 1, device=actions.device)
        loss = ((self.actor_flow(observations, (1 - t) * z + t * actions, t) - (actions - z)) ** 2).mean(dim=-1).mean()

        self.actor_flow_optimizer.zero_grad()
        loss.backward()
        self.actor_flow_optimizer.step()

        return {
            "loss": loss,
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
        mv = self.update_value(observations, actions)
        mq = self.update_q(observations, actions, rewards, next_observations, dones)
        ma = self.update_actor(observations, actions)
        self.update_target_critic()
        return {
            "v_loss": mv["loss"].item(),
            "v_mean": mv["v_mean"].item(),
            "adv_mean": mv["adv_mean"].item(),
            "q_loss": mq["q_loss"].item(),
            "q_mean": mq["q_mean"].item(),
            "q_max": mq["q_max"].item(),
            "q_min": mq["q_min"].item(),
            "actor_loss": ma["loss"].item(),
        }

    def update_target_critic(self) -> None:
        for target_param, online_param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.target_update_rate * online_param.data
                + (1 - self.target_update_rate) * target_param.data
            )
