from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List

class PQSMPDAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        inv_temp: float,
        flow_steps: int,
        rho: float, 
        num_samples: int,

    ):
        super().__init__()

        self.action_dim = action_dim

        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.inv_temp = inv_temp
        self.flow_steps = flow_steps
        self.num_samples = num_samples
        self.rho = rho

        betas = self.cosine_beta_schedule(flow_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alpha_hats", torch.cumprod(1 - betas, dim=0))

        self.to(ptu.device)
    
    def cosine_beta_schedule(self, timesteps):
        """
        Cosine annealing beta schedule
        """
        t = torch.linspace(0, timesteps, timesteps + 1, device=ptu.device) / timesteps
        f = torch.cos((t + 0.08) / 1.08 * torch.pi / 2) ** 2
        return torch.clamp(1 - f[1:] / f[:-1], 0.0001, 0.999)
    
    @torch.compiler.disable
    def ddpm_sampler(self, observations: torch.Tensor, noise: torch.Tensor):
        """
        DDPM sampling
        """
        x = noise
        for i in reversed(range(self.flow_steps)):
            t = torch.full((x.shape[0], 1), i / self.flow_steps, device=x.device)
            e = self.actor(observations, x, t)
            #had sqrt here has a possible fix, didnt work
            x = (x - self.betas[i] / torch.sqrt(1 - self.alpha_hats[i]) * e) / torch.sqrt(self.alphas[i])
            if i > 0:
                x = x + torch.sqrt(self.betas[i]) * torch.randn_like(x)
            x = torch.clamp(x, -1, 1)  # clamp each step for stability
        return x
    
    def get_action(self, observation: torch.Tensor):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        return ptu.to_numpy(self.sample_actions(observation))[0]

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
        Update Critic
        """
        with torch.no_grad():
            a = self.ddpm_sampler(next_observations, torch.randn_like(actions))
            #try mean instead of min
            target_q = self.target_critic(next_observations, a)
            y = rewards + self.discount * (1 - dones) * self.pessimistic_q(target_q)

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
    
    def pessimistic_q(self, q):
        return q.mean(dim=0) - self.rho * q.std(dim=0)
        
    @torch.compiler.disable
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the actor
        """

        z = torch.randn_like(actions)
        k = torch.randint(0, self.flow_steps, (actions.shape[0],), device=actions.device)
        a = self.alpha_hats[k][:, None]
        x = (a.sqrt() * actions + (1 - a).sqrt() * z).detach().requires_grad_(True)
        e = self.actor(observations, x, k[:, None].float() / self.flow_steps)
        # min -> mean change
        q = self.pessimistic_q(self.target_critic(observations, x)).sum()
        g = torch.autograd.grad(q, x)[0].detach()
        loss = ((-e - self.inv_temp * g) ** 2).mean() + self.alpha * ((z - e) ** 2).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": loss,
        }
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Rejection / best-of-n sampling using the flow policy and critic.

        We:
          1. Sample multiple candidate actions via the DDPM Sampler.
          2. Evaluate them with the critic.
          3. Pick the action with the highest pessimistic Q-value.
        """
        b = observations.shape[0]
        obs = observations.repeat_interleave(self.num_samples, dim=0)
        candidate = self.ddpm_sampler(obs, torch.randn(obs.shape[0], self.action_dim, device=obs.device))
        q = self.critic(obs, candidate)
        scores = self.pessimistic_q(q).view(b, self.num_samples)

        best_idx = scores.argmax(dim=1)

        return candidate.view(b, self.num_samples, self.action_dim)[torch.arange(b, device=candidate.device), best_idx]

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        mq = self.update_q(observations, actions, rewards, next_observations, dones)
        ma = self.update_actor(observations, actions)
        metrics = {
            "q_loss": mq["q_loss"].item(),
            "q_mean": mq["q_mean"].item(),
            "q_max": mq["q_max"].item(),
            "q_min": mq["q_min"].item(),
            "actor_loss": ma["actor_loss"].item(),
        }

        #logs
        with torch.no_grad():
            n = min(16, observations.shape[0])
            sample_obs = observations[:n]
            sample_act = self.ddpm_sampler(
                sample_obs,
                torch.randn(n, self.action_dim, device=sample_obs.device),
            )
            metrics["sample_abs_mean"] = sample_act.abs().mean().item()
            metrics["sample_abs_max"] = sample_act.abs().max().item()
            metrics["sample_sat_frac"] = (sample_act.abs() > 0.99).float().mean().item()
        #

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        for p, q in zip(self.target_critic.parameters(), self.critic.parameters()):
            p.data.copy_(self.target_update_rate * q.data + (1 - self.target_update_rate) * p.data)