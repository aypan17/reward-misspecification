from collections import OrderedDict

import numpy as np
import torch

import bgp.rlkit.torch.pytorch_util as ptu
from bgp.rlkit.core.eval_util import create_stats_ordered_dict
from bgp.rlkit.torch.dqn.dqn import DQN


class DoubleDQN(DQN):
    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Compute loss
        """

        best_action_idxs = self.qf(next_obs).max(
            1, keepdim=True
        )[1]
        target_q_values = self.target_qf(next_obs).gather(
            1, best_action_idxs
        ).detach()
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        if self.gradient_max_value is not None:
            torch.nn.utils.clip_grad_value_(self.qf.parameters(), self.gradient_max_value)
        self.qf_optimizer.step()
        self._update_target_network()

        """
        Save some statistics for eval using just one batch.
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Y Predictions',
                ptu.get_numpy(y_pred),
            ))
            grads = torch.tensor([], device=self.qf.device)
            for param in self.qf.parameters():
                try:
                    grads = torch.cat((grads, torch.abs(param.grad.data.flatten())))
                except:
                    pass  # seems to be a weird error on mld5 around layernorm
            self.eval_statistics['Gradient'] = grads.mean().item()
