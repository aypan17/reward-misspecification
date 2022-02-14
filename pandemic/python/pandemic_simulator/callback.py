from stable_baselines3.common.callbacks import BaseCallback
from pandemic_simulator.environment.interfaces import sorted_infection_summary

import os
import wandb
import numpy as np

class WandbCallback(BaseCallback):
	"""
	A wandb logging callback that derives from ``BaseCallback``.

	:param verbose: (int) Verbosity level 0: not output 1: info 2: debug
	"""
	def __init__(self, name="", gamma=0.99, viz=None, eval_freq=10, multiprocessing=False, log_dir="pandemic_results"):
		
		self.name = name
		self.gamma = gamma
		self.viz = viz
		self.eval_freq = eval_freq
		self.multi = multiprocessing
		self.log_dir = log_dir

		self.n_rollouts=0
		self.record = False
		super(WandbCallback, self).__init__(verbose)

	def _on_rollout_start(self) -> None:
		"""
		A rollout is the collection of environment interaction
		using the current policy.
		This event is triggered before collecting new samples.
		"""
		self.episode_rewards = []
		self.episode_reward_std = []
		self.episode_true_rewards = []
		self.episode_true_reward_std = []
		self.episode_infection_data = np.array([[0, 0, 0, 0, 0]])
		self.episode_threshold = []

		self.n_rollouts += 1
		self.record = (self.viz is not None and self.n_rollouts % self.eval_freq == 0)
		self.counter = 0

	def _on_step(self) -> bool:
		"""
		This method will be called by the model after each call to `env.step()`.

		For child callback (of an `EventCallback`), this will be called
		when the event is triggered.

		:return: (bool) If the callback returns False, training is aborted early.
		"""
		list_obs = self.training_env.get_attr("observation")
		rew = self.training_env.get_attr("last_reward")
		true_rew = self.training_env.get_attr("get_true_reward")
		infection_data = np.zeros((1, 5))
		threshold_data = np.zeros(len(list_obs))
		for obs in list_obs:
			infection_data += obs.global_infection_summary[-1]
			threshold_data += obs.infection_above_threshold[-1].item()

		self.episode_rewards.append(np.mean(rew))
		self.episode_reward_std.append(np.std(rew))
		self.episode_true_rewards.append(np.mean(true_rew))
		self.episode_true_reward_std.append(np.std(true_rew))
		self.episode_infection_data = np.concatenate([self.episode_infection_data, infection_data / len(list_obs)])
		self.episode_threshold.append(np.sum(threshold_data) / len(list_obs))
		
		if self.record and self.counter < 192:
			gis = np.array([obs.global_infection_summary[-1] for obs in list_obs]).squeeze(1)
			gts = np.array([obs.global_testing_summary[-1] for obs in list_obs]).squeeze(1)
			stage = np.array([obs.stage[-1].item() for obs in list_obs])
			self.viz.record_list(obs, gis, gts, stage, rew, true_rew)
			self.counter += 1
		return True

	def _on_rollout_end(self) -> None:
		"""
		This event is triggered before updating the policy.
		"""
		infection_summary = np.sum(self.episode_infection_data, axis=0)
		horizon = len(self.episode_rewards)
		true_w = np.geomspace(1, 1, num=horizon)
		proxy_w = np.geomspace(1, 1, num=horizon)
		n_ppl = np.sum(self.episode_infection_data[1])

		wandb.log({"reward": np.dot(proxy_w, np.array(self.episode_rewards)),
				   "reward_std": np.mean(self.episode_reward_std), 
				   "true_reward": np.dot(true_w, np.array(self.episode_true_rewards)),
				   "true_reward_std": np.mean(self.episode_true_reward_std),
				   "proportion_critical": infection_summary[0] / n_ppl,
				   "proportion_dead": infection_summary[1] / n_ppl,
				   "proportion_infected": infection_summary[2] / n_ppl,
				   "proportion_healthy": infection_summary[3] / n_ppl,
				   "proportion_recovered": infection_summary[4] / n_ppl,
				   "time_over_threshold": np.mean(self.episode_threshold),
				   })
		
		self.model.save(os.path.join(self.log_dir, self.name+"_epoch_"+str(self.n_rollouts)))
		if self.record:
			self.viz.plot(name=self.name, epoch=self.n_rollouts)
			self.viz.reset()
		print(f"{self.n_rollouts} epochs completed")

