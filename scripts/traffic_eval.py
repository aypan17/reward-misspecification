"""Plot rewards vs. norms.

Attributes
----------
EXAMPLE_USAGE : str
	Example call to the function, which is
	::

		python ./visualizer_rllib.py /tmp/ray/result_dir 1

parser : ArgumentParser
	Command-line argument parser
"""

import argparse
import gym
import numpy as np
import os
import sys
import time
from copy import deepcopy
import json
import pandas as pd

import logging

import seaborn
import scipy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as MVN
from scipy.stats.stats import pearsonr   

import ray
try:
	from ray.rllib.agents.agent import get_agent_class
except ImportError:
	from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl
from flow.core.rewards import REWARD_REGISTRY

import tensorflow as tf

logger = logging.getLogger(__name__)


EXAMPLE_USAGE="""
example usage:
	python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""

class DiagGaussian(object):
	"""Action distribution where each vector element is a gaussian.

	The first half of the input vector defines the gaussian means, and the
	second half the gaussian standard deviations.
	"""

	def __init__(self, inputs):
		mean, log_std = np.split(inputs, 2)
		self.mean = mean
		self.log_std = log_std
		self.std = np.exp(log_std)

	def kl(self, other):
		if other is None:
			return 0
		assert isinstance(other, DiagGaussian)
		if other.mean.shape != self.mean.shape:
			return None
		return np.sum(
			other.log_std - self.log_std +
			(np.square(self.std) + np.square(self.mean - other.mean)) /
			(2.0 * np.square(other.std)))

	@property
	def entropy(self):
		return np.sum(
			self.log_std + .5 * np.log(2.0 * np.pi * np.e))

def distributions_js(distribution_p, distribution_q, n_samples=10 ** 5):
	# jensen shannon divergence. (Jensen shannon distance is the square root of the divergence)
	# all the logarithms are defined as log2 (because of information entrophy)
	X = distribution_p.rvs(n_samples)
	p_X = distribution_p.pdf(X)
	q_X = distribution_q.pdf(X)
	log_mix_X = np.log2(p_X + q_X)

	Y = distribution_q.rvs(n_samples)
	p_Y = distribution_p.pdf(Y)
	q_Y = distribution_q.pdf(Y)
	log_mix_Y = np.log2(p_Y + q_Y)

	return (np.log2(p_X).mean() - (log_mix_X.mean() - np.log2(2))
			+ np.log2(q_Y).mean() - (log_mix_Y.mean() - np.log2(2))) / 2

def get_dist_params(agent_logits, base_logits):
	mean_agent, std_agent = np.split(agent_logits, 2)
	mean_base, std_base = np.split(base_logits, 2)
	cars = len(std_agent)
	cov_agent = np.zeros((cars, cars), float)
	cov_base = np.zeros((cars, cars), float)
	np.fill_diagonal(cov_agent, np.exp(std_agent))
	np.fill_diagonal(cov_base, np.exp(std_base))
	return mean_agent, cov_agent, mean_base, cov_base

def hellinger(agent_logits, base_logits):
	mu1, sigma1, mu2, sigma2 = get_dist_params(agent_logits, base_logits)
	sigma1_plus_sigma2 = sigma1 + sigma2
	mu1_minus_mu2 = mu1 - mu2
	E = mu1_minus_mu2.T @ np.linalg.inv(sigma1_plus_sigma2/2) @ mu1_minus_mu2
	epsilon = -0.125*E
	numerator = np.sqrt(np.linalg.det(sigma1 @ sigma2))
	denominator = np.linalg.det(sigma1_plus_sigma2/2)
	squared_hellinger = 1 - np.sqrt(numerator/denominator)*np.exp(epsilon)
	squared_hellinger = squared_hellinger.item()
	return np.sqrt(squared_hellinger)

def jensen_shannon(agent_logits, base_logits, n_samples=10 ** 5):
	mean_agent, cov_agent, mean_base, cov_base = get_dist_params(agent_logits, base_logits)
	agent = MVN(mean=mean_agent, cov=cov_agent)
	base = MVN(mean=mean_base, cov=cov_base)
	return distributions_js(base, agent, n_samples=n_samples)

def safe_mean(arr):
	mlen = min([len(e) for e in arr])
	return np.mean([e[:mlen] for e in arr], axis=0)


def rollout(env, args, agent, baseline_agent, true_specification, true2_specification=None):
	full_reward = []
	full_true_reward = []
	full_true_reward2 = []
	# Simulate and collect metrics
	rets = []
	true_rets = []
	true_rets2 = []
	#actions = []
	log_probs = []
	base_log_probs = []
	vfs = []
	base_vfs = []
	kls = []
	car_kls = []
	js = []
	car_js = []
	h = []
	car_h = []

	for i in range(args.num_rollouts):
		ret = 0
		true_ret = 0
		true_ret2 = 0
		#action_moments = [] 
		log_prob = []
		base_log_prob = []
		vf = []
		base_vf = []
		kl = []
		car_kl = []
		js_dist = []
		car_js_dist = []
		h_dist = []
		car_h_dist = []

		state = env.reset()
		for j in range(args.horizon):
			action = agent.compute_action(state, full_fetch=True)
			baseline_action = baseline_agent.compute_action(state, full_fetch=True)

			vf_preds = action[2]['vf_preds']
			logp = action[2]['action_logp']
			logits = action[2]['behaviour_logits']
			base_vf_preds = baseline_action[2]['vf_preds']
			base_logp = baseline_action[2]['action_logp']
			base_logits = baseline_action[2]['behaviour_logits']

			action = action[0]

			cars = []
			car_logits = []
			car_base_logits = []
#			for i, rl_id in enumerate(env.unwrapped.rl_veh):
#				# get rl vehicles inside the network
#				if rl_id in env.unwrapped.k.vehicle.get_rl_ids():
#					cars.append(i)
#			for c in cars:
#				car_logits.append(logits[c])
#				car_base_logits.append(base_logits[c])
#			for c in cars:
#				car_logits.append(logits[c + len(logits)//2])
#				car_base_logits.append(base_logits[c])
#			car_logits = np.array(car_logits)
#			car_base_logits = np.array(car_base_logits)
			
			if (j+1) % 20 == 0:
				vf.append(vf_preds)
				log_prob.append(logp)
				action_dist = DiagGaussian(logits)
				base_log_prob.append(base_logp)
				base_vf.append(base_vf_preds)
				base_action_dist = DiagGaussian(base_logits)
				kl.append(base_action_dist.kl(action_dist))
				js_dist.append(jensen_shannon(logits, base_logits))
				h_dist.append(hellinger(logits, base_logits))

				if len(cars) > 0:
					car_action_dist = DiagGaussian(car_logits)
					car_base_action_dist = DiagGaussian(car_base_logits)
					car_kl.append(car_base_action_dist.kl(car_action_dist))
					car_js_dist.append(jensen_shannon(car_logits, car_base_logits))
					car_h_dist.append(hellinger(car_logits, car_base_logits))

			state, reward, done, _ = env.step(action)
			ret += reward
			vels = np.array([env.unwrapped.k.vehicle.get_speed(veh_id) for veh_id in env.unwrapped.k.vehicle.get_ids()])
			if all(vels > -100):
				full_reward.append(reward)
				true_reward = sum([eta * REWARD_REGISTRY[rew](env, action) for rew, eta in true_specification])
				full_true_reward.append(true_reward)
				true_ret += true_reward
				if true2_specification:
					true_reward2 = sum([eta * REWARD_REGISTRY[rew](env, action) for rew, eta in true2_specification])
					full_true_reward2.append(true_reward2)
					true_ret2 += true_reward2

			if done:
				break
		if done and (j+1) != args.horizon:
			continue
		rets.append(ret)
		true_rets.append(true_ret)
		true_rets2.append(true_ret2)
		#actions.append(action_moments)
		base_log_probs.append(base_log_prob)
		log_probs.append(log_prob)
		vfs.append(vf)
		base_vfs.append(base_vf)
		kls.append(kl)
		car_kls.append(car_kl)
		js.append(js_dist)
		car_js.append(car_js_dist)
		h.append(h_dist)
		car_h.append(car_h_dist)
	logger.info('==== Finished epoch ====')
	if len(rets) == 0:
		logger.info("ERROR")
		return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
	return np.mean(rets), np.mean(true_rets), np.mean(true_rets2), \
		np.std(rets), np.std(true_rets), np.std(true_rets2), \
		   safe_mean(log_probs), safe_mean(base_log_probs), \
		   safe_mean(vfs), safe_mean(base_vfs), \
		   safe_mean(kls), safe_mean(car_kls), \
		   safe_mean(js), safe_mean(car_js), \
		   safe_mean(h), safe_mean(car_h), \
		   pearsonr(full_reward, full_true_reward), pearsonr(full_reward, full_true_reward2)

def reward_specification(rewards, weights):
	rewards = rewards.split(",")
	weights = weights.split(",")
	assert len(rewards) == len(weights)
	return [(r, float(w)) for r, w in zip(rewards, weights)]

def compute_norms(args):
	results = args.results if args.results[-1] != '/' \
		else args.results[:-1]

	params = []
	l_1 = []
	l_2 = []
	lc = []
	rew = []
	true_rew = []
	true_rew2 = []
	rs = []
	trs = []
	trs2 = []
	log_probs = []
	base_log_probs = []
	vfs = []
	base_vfs = []
	kls = []
	car_kls = []
	js = []
	car_js = []
	h = []
	car_h = []
	e = []
	m = []
	c1 = []
	c2 = []
	not_created = True

	proxy_specification = reward_specification(args.proxy, args.proxy_weights)
	true_specification = reward_specification(args.true, args.true_weights)

	if args.true2 and args.true2_weights:
		true2_specification = reward_specification(args.true2, args.true2_weights)
	else:
		true2_specification = None

	for directory in os.listdir(results):
		# misspecification = float(directory.split("_")[-1])
		misspecification = []
		#for d in os.listdir(results+'/'+directory):
		result_dir = results + '/' + directory #+ '/' + d
		if not os.path.isdir(result_dir):
			continue 
		try:
			config = get_rllib_config(result_dir)
		except:
			print(f"Loading {result_dir} config failed")
			continue
		print(result_dir)

		# Get the proxy reward at all the epochs
		if args.skip != -1:
			epochs = [str(i) for i in range(args.low, args.high+1, args.skip)]
			logger.info(f'User Defined Epochs: {epochs}')
		else:
			try:
				data = pd.read_csv(os.path.join(result_dir, 'progress.csv'))
			except:
				logger.info("CORRUPTED DATA")
				continue
			proxy = data['episode_reward_mean'].to_numpy(dtype=float)
			steps = data['training_iteration'].to_numpy(dtype=int)
			idx = [i for i in range(len(steps)) if (steps[i] % 50) == 0]
			proxy = proxy[idx]
			steps = steps[idx]
			if len(proxy) == 0:
				continue
			max_idx = np.argmax(proxy)
			last_idx = -1
			logger.info(f'Max proxy of {proxy[max_idx]} achieved at epoch {steps[max_idx]}.')
			logger.info(f'Last proxy of {proxy[last_idx]} achieved at epoch {steps[last_idx]}.')
			epochs = [steps[max_idx], 50]

		# Run on only one cpu for rendering purposes
		config['num_workers'] = 0

		flow_params = get_flow_params(config)

		# hack for old pkl files
		sim_params = flow_params['sim']
		setattr(sim_params, 'num_clients', 1)

		# for hacks for old pkl files 
		if not hasattr(sim_params, 'use_ballistic'):
			sim_params.use_ballistic = False

		# Determine agent and checkpoint
		config_run = config['env_config']['run'] if 'run' in config['env_config'] \
			else None
		if args.run and config_run:
			if args.run != config_run:
				print('visualizer_rllib.py: error: run argument '
					  + '\'{}\' passed in '.format(args.run)
					  + 'differs from the one stored in params.json '
					  + '\'{}\''.format(config_run))
				sys.exit(1)
		if args.run:
			agent_cls = get_agent_class(args.run)
		elif config_run:
			agent_cls = get_agent_class(config_run)
		else:
			print('visualizer_rllib.py: error: could not find flow parameter '
				  '\'run\' in params.json, '
				  'add argument --run to provide the algorithm or model used '
				  'to train the results\n e.g. '
				  'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
			sys.exit(1)

		sim_params.restart_instance = True
		dir_path = os.path.dirname(os.path.realpath(__file__))

		# Create and register a gym+rllib env
		create_env, env_name = make_create_env(
			params=flow_params, reward_specification=proxy_specification)
		register_env(env_name, create_env)
		create_env2, env_name2 = make_create_env(
			params=flow_params, reward_specification=proxy_specification)
		register_env(env_name2, create_env2)

		# Start the environment with the gui turned on and a path for the
		# emission file
		env_params = flow_params['env']
		env_params.restart_instance = False

		# lower the horizon if testing
		if args.horizon:
			config['horizon'] = args.horizon
			env_params.horizon = args.horizon

		# create the agent that will be used to compute the actions
		del config['callbacks']

		agent = agent_cls(env=env_name, config=config)
		if args.baseline:
			if not_created:
				try:
					config2 = get_rllib_config(args.baseline)
				except:
					logger.info(f"###### Loading baseline agent config failed ######")
					break
				del config2['callbacks']
				baseline_agent = agent_cls(env=env_name2, config=config2)
				data = pd.read_csv(os.path.join(args.baseline, 'progress.csv'))
				steps = data['training_iteration'].to_numpy(dtype=int)
				idx = [i for i in range(len(steps)) if (steps[i] % 50) == 0]
				epoch = str(steps[idx[-1]])
				checkpoint = 
					os.path.join(args.baseline, f'checkpoint_{epoch}/checkpoint-{epoch}')
				baseline_agent.restore(checkpoint)
				not_created = False
				logger.info("====== Using baseline agent ======")
		else:
			assert False
			if not not_created:
				assert False
			baseline_agent = None

		if hasattr(agent, "local_evaluator") and os.environ.get("TEST_FLAG") != 'True':
			env = agent.local_evaluator.env
		else:
			env = gym.make(env_name)

		# if restart_instance, don't restart here because env.reset will restart later
		if not sim_params.restart_instance:
			env.restart_simulation(sim_params=sim_params, render=sim_params.render)

		weights = [w for _, w in agent.get_weights()['default_policy'].items()]
		names = [k for k, _ in agent.get_weights()['default_policy'].items()]
		sizes = [w.shape for w in weights[::4]]
		p = np.sum([np.prod(s) for s in sizes]).item()
		print(p, sizes)
		for epoch in epochs:
			epoch = str(epoch)
			checkpoint = result_dir + '/checkpoint_' + epoch
			checkpoint = checkpoint + '/checkpoint-' + epoch
			if not os.path.isfile(checkpoint):
				logger.info("MISSING CHECKPOINT")
				break
			agent.restore(checkpoint)

			r, tr, tr2, rstd, trstd, trstd2, \
			logp, base_logp, vf, base_vf, kl, car_kl, js_dist, car_js_dist, \
			h_dist, car_h_dist, corr_proxy_true, corr_proxy_true2 = \
				rollout(
					env, args, agent, baseline_agent, 
					true_specification, true2_specification=true2_specification
				)
			if r is None:
				continue
			
			params.append(p)
			rew.append(r)
			true_rew.append(tr)
			true_rew2.append(tr2)
			rs.append(rstd)
			trs.append(trstd)
			trs2.append(trstd2)
			log_probs.append(logp.tolist())
			base_log_probs.append(base_logp.tolist())
			vfs.append(vf.tolist())
			base_vfs.append(vf.tolist())
			kls.append(kl.tolist())
			car_kls.append(car_kl.tolist())
			js.append(js_dist.tolist())
			car_js.append(car_js_dist.tolist())
			h.append(h_dist.tolist())
			car_h.append(car_h_dist.tolist())
			e.append(epoch)
			c1.append(corr_proxy_true)	
			c2.append(corr_proxy_true2)

		# terminate the environment
		env.unwrapped.terminate()


	with open(f'{results}_correlation.json', 'a', encoding='utf-8') as f:
		json.dump({'m': m, 'e': e, 'params': params, 
					'rew': rew, 'true_rew': true_rew, 'true_rew2': true_rew2,
					'rs': rs, 'trs': trs, 'trs2': trs2,
					'log_probs': log_probs, 'base_log_probs': base_log_probs, 
					'vfs': vfs, 'base_vfs': base_vfs, 
					'kls': kls, 'car_kls': car_kls, 
					'js': js, 'car_js': car_js, 
					'h': h, 'car_h': car_h, 'c1': c1, 'c2': c2}, f)
	f.close()
	   

def create_parser():
	"""Create the parser to capture CLI arguments."""
	parser = argparse.ArgumentParser(
		formatter_class=argparse.RawDescriptionHelpFormatter,
		description='[Flow] Evaluates a reinforcement learning agent '
					'given a checkpoint.',
		epilog=EXAMPLE_USAGE)

	# required input parameters
	parser.add_argument(
		'results', type=str, help='File with list of directory containing results')
	parser.add_argument(
		'proxy', type=str, help='Proxy reward functions to include'
	)
	parser.add_argument(
		'proxy_weights', type=str, help='Weights for proxy rewards'
	)
	parser.add_argument(
		'true', type=str, help='True reward functions to include'
	)
	parser.add_argument(
		'true_weights', type=str, help='Weights for true rewards'
	)

	# Optional inputs
	parser.add_argument(
		'--true2', type=str, default=None, help='True reward functions to include'
	)
	parser.add_argument(
		'--true2_weights', type=str, default=None, help='Weights for proxy rewards'
	)
	parser.add_argument(
		'--run',
		type=str,
		help='The algorithm or model to train. This may refer to '
			 'the name of a built-on algorithm (e.g. RLLib\'s DQN '
			 'or PPO), or a user-defined trainable function or '
			 'class registered in the tune registry. '
			 'Required for results trained with flow-0.2.0 and before.')
	parser.add_argument(
		'--num_rollouts',
		type=int,
		default=4,
		help='The number of rollouts to visualize.')
	parser.add_argument(
		'--horizon',
		default=270,
		type=int,
		help='Specifies the horizon.')
	parser.add_argument('--low', type=int, default=500, help='the epoch to start plotting from')
	parser.add_argument('--high', type=int, default=5000, help='the epoch to stop plotting from')
	parser.add_argument('--skip', type=int, default=-1, help='the epoch to stop plotting at')
	parser.add_argument('--baseline', type=str, default=None, help="the path of the trusted model for anomaly detection")
	return parser


if __name__ == '__main__':
	parser = create_parser()
	args = parser.parse_args()
	ray.init(num_cpus=1, log_to_driver=False)
	compute_norms(args)
