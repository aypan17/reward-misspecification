import time
import sys
import json 
import argparse
from tqdm import trange
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from scipy.spatial.distance import jensenshannon
import gym

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

import pandemic_simulator as ps
from pandemic_simulator.environment.reward import RewardFunction, SumReward, RewardFunctionFactory, RewardFunctionType
from pandemic_simulator.environment.interfaces import InfectionSummary
from pandemic_simulator.viz import PandemicViz
from pandemic_simulator.environment import PandemicSimOpts

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


def hellinger(p, q):
  # distance between p and q
  # p and q are np array probability distributions
  return (1.0 / np.sqrt(2.0)) * np.sqrt(np.sum(np.square(np.sqrt(p) - np.sqrt(q)), axis=1))


def evaluate_policy(
    name: str,
    model: "base_class.BaseAlgorithm",
    base_model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 32,
    deterministic: bool = True,
    render: bool = False,
    viz: Optional[PandemicViz] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    episode_rewards = []
    reward_std = []
    episode_true_rewards = []
    true_reward_std = []
    episode_true_rewards2 = []
    true_reward_std2 = []

    vfs = []
    log_probs = []
    ents = []
    base_vfs = []
    base_log_probs = []
    base_ents = []

    kls = []
    js = []
    h = []

    numpy_obs = env.reset()

    states = None
    for t in range(200):
        actions, states = model.predict(numpy_obs, state=states, deterministic=True)

        vf, logp, ent = model.policy.evaluate_actions(torch.as_tensor(numpy_obs), torch.as_tensor(actions))
        base_vf, base_logp, base_ent = base_model.policy.evaluate_actions(torch.as_tensor(numpy_obs), torch.as_tensor(actions))

        vfs.append(torch.mean(vf).detach().item())
        log_probs.append(torch.mean(logp).detach().item())
        ents.append(torch.mean(ent).detach().item())
        base_vfs.append(torch.mean(base_vf).detach().item())
        base_log_probs.append(torch.mean(base_logp).detach().item())
        base_ents.append(torch.mean(base_ent).detach().item())

        # Distances
        log_ratio = logp - base_logp
        # Estimator of KL from http://joschu.net/blog/kl-approx.html
        kls.append(torch.mean(torch.exp(log_ratio) - 1 - log_ratio).item())

        latent_pi, _, latent_sde = model.policy._get_latent(torch.as_tensor(numpy_obs))
        model_dist = model.policy._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde).distribution.probs.detach().numpy()
        latent_pi, _, latent_sde = base_model.policy._get_latent(torch.as_tensor(numpy_obs))
        base_dist = base_model.policy._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde).distribution.probs.detach().numpy()
        js.append(np.mean(jensenshannon(model_dist, base_dist, axis=1)).item())
        h.append(np.mean(hellinger(model_dist, base_dist)).item())

        numpy_obs, _, done, info = env.step(actions)
        
        rew = env.get_attr("last_reward")
        true_rew = env.get_attr("get_true_reward")
        true_rew2 = env.get_attr("get_true_reward2")
        episode_rewards.append(np.mean(rew))
        reward_std.append(rew)
        episode_true_rewards.append(np.mean(true_rew))
        true_reward_std.append(true_rew)
        episode_true_rewards2.append(np.mean(true_rew2))
        true_reward_std2.append(true_rew2)

        obs = env.get_attr("observation")
        infection_data = np.zeros((1, 5))
        threshold_data = np.zeros(len(obs))
        for o in obs:
            infection_data += o.global_infection_summary[-1]

        gis = np.array([o.global_infection_summary[-1] for o in obs]).squeeze(1)
        gts = np.array([o.global_testing_summary[-1] for o in obs]).squeeze(1)
        stage = np.array([o.stage[-1].item() for o in obs])
        if viz:
            viz.record_list(obs[0], gis, gts, stage, rew, true_rew, true_rew2=true_rew2)

    reward = np.sum(episode_rewards).item()
    true_reward = np.sum(episode_true_rewards).item()
    true_reward2 = np.sum(episode_true_rewards2).item()
    
    #if viz:
    #    viz.plot(name=name, evaluate=True, 
    #    plots_to_show=['critical_summary', 'stages', 'cumulative_reward', 'cumulative_true_reward2'])
    #    viz.reset()

    return reward, np.std(np.sum(np.array(reward_std), axis=0)).item(), \
            true_reward, np.std(np.sum(np.array(true_reward_std), axis=0)).item(), \
            true_reward2, np.std(np.sum(np.array(true_reward_std2), axis=0)).item(), \
            kls, js, h, log_probs, base_log_probs, vfs, base_vfs

def plot_critical_summary(ax, viz, color, sty, m):
    gis = np.vstack(viz._gis).squeeze()
    gis_std = np.vstack(viz._gis_std).squeeze()
    ax.plot(viz._num_persons * gis[:, viz._critical_index], color='black', linestyle=sty, linewidth=1, label='_nolegend_')
    # ax.fill_between(
    #   np.arange(len(gis)), 
    #   viz._num_persons * (gis-gis_std)[:, viz._critical_index], 
    #   viz._num_persons * (gis+gis_std)[:, viz._critical_index], 
    #   alpha=0.1, color=color
    # )
    ax.plot(np.arange(gis.shape[0]), np.ones(gis.shape[0]) * viz._max_hospital_capacity, 'y')
    ax.legend(['Max hospital capacity'], loc='upper left')
    ax.set_ylim(-0.1, viz._max_hospital_capacity * 3)
    ax.set_title('ICU Usage', fontsize=16)
    ax.set_xlabel('time (days)', fontsize=16)
    ax.set_ylabel('persons', fontsize=16)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    height = viz._num_persons * gis[m, viz._critical_index]
    ax.plot([m, m], [-0.1, height], color=color, linestyle=sty, linewidth=2)
    ax.plot([0, m], [height, height], color=color, linestyle=sty, linewidth=2)


def plot_stages(ax, viz, color, sty):
    days = np.arange(len(viz._stages))
    stages = np.array(viz._stages)
    stages_std = np.array(viz._stages_std)
    ax.plot(days, stages, color='black', linestyle=sty, linewidth=1)
    #ax.fill_between(days, stages - stages_std, stages + stages_std, alpha=0.1, color=color)
    ax.set_ylim(-0.1, 5) # This assumes at most 5 stages!!
    ax.set_title('Regulation Stage', fontsize=16)
    ax.set_xlabel('time (days)', fontsize=16)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    m = np.argmax(stages[50:]) + 50
    ax.plot([m, m], [-0.1, stages[m]], color=color, linestyle=sty, linewidth=2)
    p1 = Line2D([0,1],[0,1],linestyle='-', color='black')
    p2 = Line2D([0,1],[0,1],linestyle='--', color='black')
    ax.legend([p1, p2], ['smaller policy', 'larger policy'], loc='upper right')
    return m

def plot(v1, v2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    c1 = 'red'
    c2 = 'blue'
    s1 = '-'
    s2 = '--'

    m1 = plot_stages(ax2, v1, c1, s1)
    plot_critical_summary(ax1, v1, c1, s1, m1)
    m2 = plot_stages(ax2, v2, c2, s2)
    plot_critical_summary(ax1, v2, c2, s2, m2)
    ax1.figure.set_size_inches(4, 3)
    ax2.figure.set_size_inches(4, 3)
    fig.set_size_inches(8, 3)
    
    plt.savefig('test.svg',dpi=120, bbox_inches='tight', pad_inches = 0, format='svg')


def make_cfg():
    sim_config = ps.env.PandemicSimConfig(
         num_persons=500,
         location_configs=[
             ps.env.LocationConfig(ps.env.Home, num=150),
             ps.env.LocationConfig(ps.env.GroceryStore, num=2, num_assignees=5, state_opts=dict(visitor_capacity=30)),
             ps.env.LocationConfig(ps.env.Office, num=2, num_assignees=150, state_opts=dict(visitor_capacity=0)),
             ps.env.LocationConfig(ps.env.School, num=10, num_assignees=2, state_opts=dict(visitor_capacity=30)),
             ps.env.LocationConfig(ps.env.Hospital, num=1, num_assignees=15, state_opts=dict(patient_capacity=5)),
             ps.env.LocationConfig(ps.env.RetailStore, num=2, num_assignees=5, state_opts=dict(visitor_capacity=30)),
             ps.env.LocationConfig(ps.env.HairSalon, num=2, num_assignees=3, state_opts=dict(visitor_capacity=5)),
             ps.env.LocationConfig(ps.env.Restaurant, num=1, num_assignees=6, state_opts=dict(visitor_capacity=30)),
             ps.env.LocationConfig(ps.env.Bar, num=1, num_assignees=3, state_opts=dict(visitor_capacity=30))
         ],
         person_routine_assignment=ps.sh.DefaultPersonRoutineAssignment(),
	 delta_start_lo = 95,
	 delta_start_hi = 105
    )
    return sim_config

def make_reg():
    return ps.sh.austin_regulations

def make_sim(sim_config, rate):
    sim_opt = PandemicSimOpts()
    sim_opt.spontaneous_testing_rate = rate
    return ps.env.PandemicSim.from_config(sim_config=sim_config, sim_opts=sim_opt)

def make_viz(sim_config):
    return ps.viz.GymViz.from_config(sim_config=sim_config)

def load_model(env, model_path, width, depth):
    agent = ps.model.StageModel(env = env)
    d_model = width
    n_layers = depth
    net_arch = [d_model] * n_layers if n_layers != 0 else []

    policy_kwargs = {
        "net_arch": [dict(pi=net_arch, vf=net_arch)], 
    }

    model = agent.get_model("ppo", policy_kwargs = policy_kwargs, verbose = 0)
    return model.load(model_path)

def init(args, rate):
    n_cpus = args.n_cpus
    ps.init_globals(seed=args.seed)
    sim_config = make_cfg()
    regulations = make_reg()
    viz = make_viz(sim_config)
    done_fn = ps.env.DoneFunctionFactory.default(ps.env.DoneFunctionType.TIME_LIMIT, horizon=200)

    reward_fn = SumReward(
            reward_fns=[
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=sim_config.max_hospital_capacity / sim_config.num_persons),
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABSOLUTE,
                                              summary_type=InfectionSummary.CRITICAL),
                RewardFunctionFactory.default(RewardFunctionType.LOWER_STAGE,
                                              num_stages=len(regulations)),
                RewardFunctionFactory.default(RewardFunctionType.SMOOTH_STAGE_CHANGES,
                                              num_stages=len(regulations))
            ],
            weights=[0, 10, 0.1, 0.01]
        )

    gym = ps.env.PandemicPolicyGymEnv.from_config(
            sim_config=sim_config, 
            sim_opts = PandemicSimOpts(spontaneous_testing_rate=rate),
            pandemic_regulations=regulations, 
            done_fn=done_fn,
            reward_fn=reward_fn,
            constrain=True,
            four_start=False,
            obs_history_size=3,
            num_days_in_obs=8
        )
    env = gym.get_multi_env(n=n_cpus) if n_cpus > 1 else gym.get_single_env()
    return env, viz

def evaluate(env, model_path, width, depth, base_model, viz):
    model = load_model(env, model_path, width, depth)
    model_parameters = filter(lambda p: p.requires_grad, model.policy.mlp_extractor.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    params = int(params)
    print(f"Evaluating {model_path+str(width)}...")
    reward, rstd, true_reward, trstd, true_reward2, tr2std, \
        kl, js, h, log_probs, base_log_probs, vfs, base_vfs = evaluate_policy(model_path, model, base_model, env, viz=viz)
    env.close()
    print(f"Model: {model_path}. Proxy: {reward}. Objective: {true_reward}.")
    return params, reward, rstd, true_reward, trstd, true_reward2, tr2std, kl, js, h, log_probs, base_log_probs, vfs, base_vfs 
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='model path')
    parser.add_argument('base_model_path', type=str, help='trusted model path')
    parser.add_argument('base_width', type=int, help='trusted model hidden units') 
    parser.add_argument('base_depth', type=int, help='trusted model layers')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--n_cpus', type=int, default=32)
    parser.add_argument('--n_episodes', type=int, default=32, help='number of episodes to evaluate (to reduce variance)')
    parser.add_argument('--epoch', type=int, default=0, help='episode to evaluate model on')
    parser.add_argument('--width', type=int, default=0, help='model hidden units')
    parser.add_argument('--rate', type=str, default="", help='spontaneous_testing_rate')
    args = parser.parse_known_args(sys.argv[1:])[0]

    params, reward, reward_std, true_reward, true_reward_std, true_reward2, true_reward2_std, kls, js, h, log_probs, base_log_probs, vfs, base_vfs, e, rates = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for w in [args.width]:
        for rate in ['01', '02', '003', '005', '03', '04', '05', '06', '07', '08', '09', '095', '1']:
            n2r = {'01':0.1, '02':0.2, '003':0.03, '005':0.05, '03':0.3, '04':0.4, '05':0.5, '06':0.6, '07':0.7, '08':0.8, '09':0.9, '095':0.95, '1':1}
            env, viz = init(args, n2r[rate])
            base_model = load_model(env, args.base_model_path, args.base_width, args.base_depth)
            p, r, rs, tr, trs, tr2, tr2s, kl, j_s, h_, logp, blogp, vf, bvf = 
                evaluate(env, args.model_path+rate+"_"+str(w), w, 2, base_model, viz)
            rates.append(n2r[rate])
            params.append(p)
            reward.append(r)
            reward_std.append(rs)
            true_reward.append(tr)
            true_reward_std.append(trs)
            true_reward2.append(tr2)
            true_reward2_std.append(tr2s)
            kls.append(kl)
            js.append(j_s)
            h.append(h_)
            log_probs.append(logp)
            base_log_probs.append(blogp)
            vfs.append(vf)
            base_vfs.append(bvf)
            e.append(args.epoch)

    f = open(f"pandemic_{args.epoch}_{args.width}_rate.json", "w")
    json.dump({
        'params':params, 'rate':rates, 'e': e,
        'rew': reward, 'rew_std': reward_std, 
        'true_rew': true_reward, 'true_rew_std': true_reward_std, 
        'true_rew2': true_reward2, 'true_rew2_std': true_reward2_std, 
        'kls': kls, 'js': js, 'h': h, 
        'log_probs': log_probs, 'base_log_probs': base_log_probs, 
        'vfs': vfs, 'base_vfs': base_vfs}, f)
    f.close()


if __name__ == '__main__':
    main()
