# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import List, Optional, Dict, Tuple, Mapping, Type, Sequence

import numpy as np
from copy import deepcopy

import gym
from gym import spaces

from .done import DoneFunction
from .interfaces import LocationID, PandemicObservation, NonEssentialBusinessLocationState, PandemicRegulation, \
	InfectionSummary, sorted_infection_summary
from .pandemic_sim import PandemicSim
from .reward import RewardFunction, SumReward, RewardFunctionFactory, RewardFunctionType
from .simulator_config import PandemicSimConfig
from .simulator_opts import PandemicSimOpts

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

__all__ = ['PandemicGymEnv', 'PandemicPolicyGymEnv']


class PandemicGymEnv(gym.Env):
	"""A gym environment interface wrapper for the Pandemic Simulator."""

	_pandemic_sim: PandemicSim
	_stage_to_regulation: Mapping[int, PandemicRegulation]
	_obs_history_size: int
	_sim_steps_per_regulation: int
	_non_essential_business_loc_ids: Optional[List[LocationID]]
	_reward_fn: Optional[RewardFunction]
	_done_fn: Optional[DoneFunction]

	_obs_with_history: np.ndarray
	_last_observation: PandemicObservation
	_last_reward: float

	def __init__(self,
				 pandemic_sim: PandemicSim,
				 pandemic_regulations: Sequence[PandemicRegulation],
				 reward_fn: Optional[RewardFunction] = None,
				 true_reward_fn: Optional[RewardFunction] = None,
				 done_fn: Optional[DoneFunction] = None,
				 obs_history_size: int = 1,
				 num_days_in_obs: int = 1,
				 sim_steps_per_regulation: int = 24,
				 non_essential_business_location_ids: Optional[List[LocationID]] = None,
				 constrain: bool = False, 
				 four_start: bool = False
				 ):
		"""
		:param pandemic_sim: Pandemic simulator instance
		:param pandemic_regulations: A sequence of pandemic regulations
		:param reward_fn: reward function
		:param done_fn: done function
		:param obs_history_size: number of latest sim step states to include in the observation
		:param sim_steps_per_regulation: number of sim_steps to run for each regulation
		:param non_essential_business_location_ids: an ordered list of non-essential business location ids
		"""
		self._pandemic_sim = pandemic_sim
		self._stage_to_regulation = {reg.stage: reg for reg in pandemic_regulations}
		self._obs_history_size = obs_history_size
		self._num_days_in_obs = num_days_in_obs
		self._sim_steps_per_regulation = sim_steps_per_regulation

		if non_essential_business_location_ids is not None:
			for loc_id in non_essential_business_location_ids:
				assert isinstance(self._pandemic_sim.state.id_to_location_state[loc_id],
								  NonEssentialBusinessLocationState)
		self._non_essential_business_loc_ids = non_essential_business_location_ids

		self._reward_fn = reward_fn
		self._true_reward_fn = true_reward_fn 
		self._done_fn = done_fn

		self._obs_with_history = self.obs_to_numpy(PandemicObservation.create_empty(history_size=self._obs_history_size*self._num_days_in_obs))
		self.observation_space = spaces.Box(
			low=0, high=np.inf, shape=self._obs_with_history.shape
		)

		self.constrain = constrain 
		if self.constrain:
			self.action_space = gym.spaces.Discrete(3) 
		else:
			self.action_space = gym.spaces.Discrete(len(self._stage_to_regulation))
		self.four_start = four_start


	@classmethod
	def from_config(cls: Type['PandemicGymEnv'],
					sim_config: PandemicSimConfig,
					pandemic_regulations: Sequence[PandemicRegulation],
					sim_opts: PandemicSimOpts = PandemicSimOpts(),
					reward_fn: Optional[RewardFunction] = None,
					done_fn: Optional[DoneFunction] = None,
					obs_history_size: int = 1,
					num_days_in_obs: int = 1,
					non_essential_business_location_ids: Optional[List[LocationID]] = None,
					) -> 'PandemicGymEnv':
		"""
		Creates an instance using config

		:param sim_config: Simulator config
		:param pandemic_regulations: A sequence of pandemic regulations
		:param sim_opts: Simulator opts
		:param reward_fn: reward function
		:param done_fn: done function
		:param obs_history_size: number of latest sim step states to include in the observation
		:param non_essential_business_location_ids: an ordered list of non-essential business location ids
		"""
		sim = PandemicSim.from_config(sim_config, sim_opts)

		if sim_config.max_hospital_capacity == -1:
			raise Exception("Nothing much to optimise if max hospital capacity is -1.")

		reward_fn = reward_fn or SumReward(
			reward_fns=[
				RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
											  summary_type=InfectionSummary.CRITICAL,
											  threshold=sim_config.max_hospital_capacity / sim_config.num_persons),
				RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
											  summary_type=InfectionSummary.CRITICAL,
											  threshold=3 * sim_config.max_hospital_capacity / sim_config.num_persons),
				RewardFunctionFactory.default(RewardFunctionType.LOWER_STAGE,
											  num_stages=len(pandemic_regulations)),
				RewardFunctionFactory.default(RewardFunctionType.SMOOTH_STAGE_CHANGES,
											  num_stages=len(pandemic_regulations))
			],
			weights=[.4, 1, .1, 0.02]
		)

		return PandemicGymEnv(pandemic_sim=sim,
							  pandemic_regulations=pandemic_regulations,
							  sim_steps_per_regulation=sim_opts.sim_steps_per_regulation,
							  reward_fn=reward_fn,
							  done_fn=done_fn,
							  obs_history_size=obs_history_size,
							  num_days_in_obs=num_days_in_obs,
							  non_essential_business_location_ids=non_essential_business_location_ids)

	@property
	def pandemic_sim(self) -> PandemicSim:
		return self._pandemic_sim

	@property
	def observation(self) -> PandemicObservation:
		return self._last_observation

	@property
	def last_reward(self) -> float:
		return self._last_reward

	@property
	def get_true_reward(self) -> float:
		return self._last_true_reward

	@property
	def get_true_reward2(self) -> float:
		return self._last_true_reward

	def obs_to_numpy(self, obs: PandemicObservation) -> np.ndarray:
		return np.concatenate([obs.time_day, obs.stage, obs.infection_above_threshold, obs.global_testing_summary_alpha, obs.global_testing_summary_delta], axis=2)

	def step(self, action: int) -> Tuple[PandemicObservation, float, bool, Dict]:
		#assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

		# execute the action if different from the current stage
		if self.constrain:
			prev_stage = self._last_observation.stage[-1, 0, 0]
			regulation = self._stage_to_regulation[int(min(max(prev_stage + action - 1, 0), 4))]
			self._pandemic_sim.impose_regulation(regulation=regulation)

		else:
			if action != self._last_observation.stage[-1, 0, 0]:  # stage has a TNC layout
				regulation = self._stage_to_regulation[action]
				self._pandemic_sim.impose_regulation(regulation=regulation)

		
		# update the sim until next regulation interval trigger and construct obs from state hist
		obs = PandemicObservation.create_empty(
			history_size=self._obs_history_size,
			num_non_essential_business=len(self._non_essential_business_loc_ids)
			if self._non_essential_business_loc_ids is not None else None)

		hist_index = 0
		for i in range(self._sim_steps_per_regulation):
			# step sim
			self._pandemic_sim.step()

			# store only the last self._history_size state values
			if (i+1) % (self._sim_steps_per_regulation // self._obs_history_size) == 0:
				obs.update_obs_with_sim_state(self._pandemic_sim.state, hist_index,
											  self._non_essential_business_loc_ids)
				hist_index += 1

			# append the last timestep if there's an overflow
			if (i+1) == self._sim_steps_per_regulation and \
				self._sim_steps_per_regulation % self._obs_history_size != 0:
					obs.update_obs_with_sim_state(self._pandemic_sim.state, hist_index,
												  self._non_essential_business_loc_ids)

		prev_obs = self._last_observation
		self._last_reward, last_rew_breakdown = self._reward_fn.calculate_reward(prev_obs, action, obs) if self._reward_fn else 0.
		self._last_true_reward, last_true_rew_breakdown = \
			self._true_reward_fn.calculate_reward(prev_obs, action, obs) if self._true_reward_fn is not None else 0.
		done = self._done_fn.calculate_done(obs, action) if self._done_fn else False
		self._last_observation = obs
		self._obs_with_history = np.concatenate([self._obs_with_history[self._obs_history_size:], self.obs_to_numpy(self._last_observation)])
		return self._obs_with_history, self._last_reward, done, {"rew": last_rew_breakdown, "true_rew": last_true_rew_breakdown}
		

	def reset(self) -> np.ndarray:
		self._pandemic_sim.reset()
		self._last_reward = 0.0
		self._last_true_reward = 0.0
		if self._done_fn is not None:
			self._done_fn.reset()

		self._last_observation = PandemicObservation.create_empty(
			history_size=self._obs_history_size,
			num_non_essential_business=len(self._non_essential_business_loc_ids)
			if self._non_essential_business_loc_ids is not None else None)
		self._obs_with_history = self.obs_to_numpy(PandemicObservation.create_empty(
			history_size=self._obs_history_size*self._num_days_in_obs,
			num_non_essential_business=len(self._non_essential_business_loc_ids)
			if self._non_essential_business_loc_ids is not None else None))

		if self.four_start:
			return self.step(4)[0]
		else:
			return self._obs_with_history

	def render(self, mode: str = 'human') -> bool:
		pass

class PandemicPolicyGymEnv(PandemicGymEnv):

	def __init__(self,
				 pandemic_sim: PandemicSim,
				 pandemic_regulations: Sequence[PandemicRegulation],
				 reward_fn: Optional[RewardFunction] = None,
				 true_reward_fn: Optional[RewardFunction] = None,
				 done_fn: Optional[DoneFunction] = None,
				 obs_history_size: int = 1,
				 num_days_in_obs: int = 1,
				 sim_steps_per_regulation: int = 24,
				 non_essential_business_location_ids: Optional[List[LocationID]] = None,
				 constrain: bool = False,
				 four_start: bool = False,
				 ):

		super().__init__(pandemic_sim,
				 pandemic_regulations,
				 reward_fn,
				 true_reward_fn,
				 done_fn,
				 obs_history_size,
				 num_days_in_obs,
				 sim_steps_per_regulation,
				 non_essential_business_location_ids,
				 constrain,
				 four_start
				)
		

	@classmethod
	def from_config(cls: Type['PandemicPolicyGymEnv'],
					sim_config: PandemicSimConfig,
					pandemic_regulations: Sequence[PandemicRegulation],
					sim_opts: PandemicSimOpts = PandemicSimOpts(),
					reward_fn: Optional[RewardFunction] = None,
					done_fn: Optional[DoneFunction] = None,
					obs_history_size: int = 1,
					num_days_in_obs: int = 1,
					non_essential_business_location_ids: Optional[List[LocationID]] = None,
					alpha: float = 0.4,
					beta: float = 1,
					gamma: float = 0.1,
					delta: float = 0.02,
					constrain: bool = False,
					four_start: bool = False
					) -> 'PandemicPolicyGymEnv':
		"""
		Creates an instance using config

		:param sim_config: Simulator config
		:param pandemic_regulations: A sequence of pandemic regulations
		:param raw_regulations: The raw regulations output by regulation_network before processing
		:param sim_opts: Simulator opts
		:param reward_fn: reward function
		:param done_fn: done function
		:param obs_history_size: number of latest sim step states to include in the observation
		:param non_essential_business_location_ids: an ordered list of non-essential business location ids
		"""
		sim = PandemicSim.from_config(sim_config, sim_opts)

		if sim_config.max_hospital_capacity == -1:
			raise Exception("Nothing much to optimise if max hospital capacity is -1.")

		reward_fn = reward_fn or SumReward(
			reward_fns=[
				RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
											  summary_type=InfectionSummary.CRITICAL,
											  threshold=sim_config.max_hospital_capacity / sim_config.num_persons),
				RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
											  summary_type=InfectionSummary.CRITICAL,
											  threshold=3 * sim_config.max_hospital_capacity / sim_config.num_persons),
				RewardFunctionFactory.default(RewardFunctionType.LOWER_STAGE,
											  num_stages=len(pandemic_regulations)),
				RewardFunctionFactory.default(RewardFunctionType.SMOOTH_STAGE_CHANGES,
											  num_stages=len(pandemic_regulations))
			],
			weights=[alpha, beta, gamma, delta]
		)

		true_reward_fn = SumReward(
			reward_fns=[
				RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABSOLUTE,
											  summary_type=InfectionSummary.CRITICAL),
				RewardFunctionFactory.default(RewardFunctionType.POLITICAL,
											   summary_type=InfectionSummary.CRITICAL),
				RewardFunctionFactory.default(RewardFunctionType.LOWER_STAGE,
											  num_stages=len(pandemic_regulations)),
				RewardFunctionFactory.default(RewardFunctionType.SMOOTH_STAGE_CHANGES,
											  num_stages=len(pandemic_regulations))
			],
			weights=[10, 10, 0.1, 0.02]
		)

		return PandemicPolicyGymEnv(pandemic_sim=sim,
							  pandemic_regulations=pandemic_regulations,
							  sim_steps_per_regulation=sim_opts.sim_steps_per_regulation,
							  reward_fn=reward_fn,
							  true_reward_fn=true_reward_fn,
							  done_fn=done_fn,
							  obs_history_size=obs_history_size,
							  non_essential_business_location_ids=non_essential_business_location_ids,
							  constrain=constrain,
							  four_start=four_start)

	def get_single_env(self):
		def get_self():
			s = deepcopy(self)
			s._pandemic_sim._numpy_rng=np.random.RandomState(0)
			return s

		e = DummyVecEnv([get_self])
		obs = e.reset()
		return e

	def get_multi_env(self, n=10):
		def get_self():
			s = deepcopy(self)
			s._pandemic_sim._numpy_rng=np.random.RandomState(np.random.randint(low=0,high=2**31))
			return s

		e = SubprocVecEnv([get_self for i in range(n)], start_method="fork")
		obs = e.reset()
		return e
