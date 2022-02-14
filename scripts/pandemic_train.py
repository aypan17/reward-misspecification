from tqdm import trange

import torch

import time
import numpy as np

import pandemic_simulator as ps
from pandemic_simulator.environment.reward import RewardFunction, SumReward, RewardFunctionFactory, RewardFunctionType
from pandemic_simulator.environment.interfaces import InfectionSummary
from pandemic_simulator.callback import WandbCallback
from pandemic_simulator.environment.simulator_opts import PandemicSimOpts
import sys
import wandb

import argparse

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def make_cfg(args):
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
         delta_start_lo = args.delta_lo,
         delta_start_hi = args.delta_hi,
    )
    return sim_config

def make_reg():
    return ps.sh.austin_regulations

def make_viz(sim_config):
    return ps.viz.GymViz.from_config(sim_config=sim_config)

def make_model(env, args, config):
    agent = ps.model.StageModel(env = env)
    ppo_params = {'n_steps': 1920, 'ent_coef': 0.01, 'learning_rate': 0.0003, 'batch_size': 64, 'gamma': 0.99}

    d_model = args.width
    n_layers = args.depth
    net_arch = [d_model] * n_layers if n_layers != 0 else []

    policy_kwargs = {
        "net_arch": [dict(pi=net_arch, vf=net_arch)], 
    }

    model = agent.get_model(
        "ppo",  
        model_kwargs = ppo_params, 
        policy_kwargs = policy_kwargs, verbose = 0
    )
    return model

def init(args):
    cfg = wandb.config
    n_cpus = args.n_cpus
    ps.init_globals(seed=0)
    sim_config = make_cfg(args)
    regulations = make_reg()
    viz = make_viz(sim_config)
    done_fn = ps.env.DoneFunctionFactory.default(ps.env.DoneFunctionType.TIME_LIMIT, horizon=193 if cfg.four_start else 192)

    # Adjust these rewards as desired
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
            weights=[float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])]
        )

    sim_opt = PandemicSimOpts(spontaneous_testing_rate=args.testing_rate)
    sim_opt.spontaneous_testing_rate = float(sys.argv[10])
    gym = ps.env.PandemicPolicyGymEnv.from_config(
            sim_config=sim_config,
            sim_opts=sim_opt, 
            pandemic_regulations=regulations, 
            done_fn=done_fn,
            reward_fn=reward_fn,
            constrain=True,
            four_start=cfg.four_start,
            obs_history_size=3,
            num_days_in_obs=8
        )
    env = gym.get_multi_env(n=n_cpus) if n_cpus > 1 else gym.get_single_env()
    return env, gym.get_single_env(), viz

def train(env, test_env, viz, args, config):
    model = make_model(env, args, config)
    if args.test:
        model.learn(
            total_timesteps = 384, 
            callback = WandbCallback(name=args.name, gamma=0.99, viz=viz, multiprocessing=(args.n_cpus>1))
        )
    else:
        model.learn(
            total_timesteps = 3072 * 5000, 
            callback = WandbCallback(name=args.name, gamma=0.99, viz=viz, multiprocessing=(args.n_cpus>1))
        )   

def main():
    parser = argparse.ArgumentParser()
    # Experiment parameters
    parser.add_argument('--name', type=str, default="pandemic",
        help='experiment name')
    parser.add_argument('--seed', type=int, default=0,
        help='random seed')
    parser.add_argument('--n_cpus', type=int, default=32,
        help='number of parallel environments to run; at least 16 is recommended')
    parser.add_argument('--log_dir', type=str, default="pandemic_results",
        help='where to store the trained models')
    parser.add_argument('--test', action='store_true',
        help='run a test experiment')

    # Model parameters
    parser.add_argument('--width', type=int, default=64,
        help='number of hidden units in the policy model')
    parser.add_argument('--depth', type=int, default=2,
        help='number of layers in the policy model')

    # Environment parameters
    parser.add_argument('--four_start', action='store_true',
        help='start the simulator at maximum regulation stage')
    parser.add_argument('--testing_rate', type=float, default=0.02,
        help='spontaneous testing rate; the higher the rate, the closer the obs reflect reality')
    """
        A random day DELTA will be selected uniformly at random from the interval [delta_lo, delta_hi].
        On day DELTA, the second wave of infections will begin. If DELTA does not lie in the episode,
        the second wave will not occur.
    """
    parser.add_argument('--delta_lo', type=int, default=95)
    parser.add_argument('--delta_hi', type=int, default=105)

    args = parser.parse_known_args(sys.argv[1:])[0]

    if args.test:
        wandb.init(
          project='TEST_PROJECT',
          group="covid",
          entity='ENTITY',
          config=cfg,
          sync_tensorboard=True
        )
    else:
        wandb.init(
          project='PROJECT',
          group="covid",
          entity='ENTITY',
          config=cfg,
          sync_tensorboard=True
        )
    train_env, test_env, viz = init(args)
    train(train_env, test_env, viz, args, config)


if __name__ == '__main__':
    main()
