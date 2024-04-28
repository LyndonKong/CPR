from operator import itemgetter
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from collections import deque

from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic
from src.policy import TD3Policy
from offlinerllib.utils.misc import convert_to_tensor, make_target
from offlinerllib.utils.noise import WhiteNoise


class CPRPolicy(TD3Policy):
    def __init__(
        self, 
        actor: BaseActor, 
        critic: Critic, 
        actor_optim: torch.optim.Optimizer, 
        critic_optim: torch.optim.Optimizer, 
        alpha: float = 0.2, 
        actor_update_interval: int = 2, 
        policy_noise: float = 0.2, 
        noise_clip: float = 0.5, 
        tau: float = 0.005, 
        discount: float = 0.99, 
        max_action: float = 1.0, 
        device: Union[str, torch.device] = "cpu",
        eta: float=1,
    ) -> None:
        super().__init__(
            actor=actor, 
            critic=critic, 
            actor_optim=actor_optim, 
            critic_optim=critic_optim, 
            actor_update_interval=actor_update_interval, 
            policy_noise=policy_noise, 
            noise_clip=noise_clip, 
            exploration_noise=WhiteNoise(mu=0, sigma=0.2), 
            tau=tau, 
            discount=discount, 
            max_action=max_action, 
            device=device
        )
        self.alpha = alpha
        self.eta = eta
        self.chords = deque(maxlen=5)


    @torch.no_grad()
    def select_action(
        self, 
        obs: np.ndarray, 
        deterministic: bool=False
    ) -> np.ndarray:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        candidate_actions = []
        action, *_ = self.actor.sample(obs, deterministic)
        if not deterministic and self._exploration_noise is not None:
            action = action.cpu().numpy()
            action = np.clip(action + self._exploration_noise(action.shape), -self._max_action, self._max_action)
        action = torch.tensor(action, requires_grad=False).to(self.device)
        candidate_actions.append(action)
        selected_action = action
        if len(self.chords) >= 2:
            for actor in self.chords:
                action, *_ = actor.sample(obs, deterministic)
                candidate_actions.append(action)
            obss = obs.repeat((len(candidate_actions), 1))
            actions = torch.concat(candidate_actions, dim=0).to(self.device)
            q = self.critic(
                obss,
                actions
            ).min(0)[0].squeeze()
            weights = torch.nn.functional.softmax(q/self.eta, dim=-1)
            select_index = torch.multinomial(weights, 1).item()
            selected_action = actions[select_index, :]
        return selected_action.squeeze().cpu().numpy()

    def actor_revitalize(self):
        archived_actor = deepcopy(self.actor)
        self.chords.append(archived_actor.eval())
        # since the actor network is small, reset all layers
        for layer in self.actor.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.actor_target.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.actor_target = make_target(self.actor)

    def offline_actor_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        obss, actions = itemgetter("observations", "actions")(batch)
        new_actions, *_ = self.actor.sample(obss)
        new_q1 = self.critic(obss, new_actions)[0, ...]
        bc_loss = F.mse_loss(new_actions, actions)
        q_loss = -self.alpha / (new_q1.abs().mean().detach()) * new_q1.mean()
        total_loss = bc_loss + q_loss
        return q_loss+bc_loss, {
            "loss/actor_bc_loss": bc_loss.item(), 
            "loss/actor_q_loss": q_loss.item(), 
            "loss/actor_total_loss": total_loss.item()
        }

    def update(
        self,
        batch: Dict[str, Any],
        stage="finetune"
    ) -> Dict[str, float]:
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        metrics = {}
        self._update_cnt += 1

        if stage in ["finetune", "alignment"]:
            critic_loss, critic_loss_metrics = self.critic_loss(batch)
            metrics.update(critic_loss_metrics)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        actor_loss = None
        if stage == "finetune":
            if (self._update_cnt % self._actor_update_interval == 0):
                actor_loss, actor_loss_metrics = self.offline_actor_loss(batch)
        if stage == "revitalization":
            actor_loss, actor_loss_metrics = self.offline_actor_loss(batch)
        if actor_loss is not None:
            metrics.update(actor_loss_metrics)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
        self._sync_weight()
        return metrics
