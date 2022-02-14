# Reward misspecification notes
## Installation
* Follow the instructions for installing flow from [here](https://flow.readthedocs.io/en/latest/flow_setup.html#installing-flow-and-sumo). 
* When cloning the `flow/` repo, replace the `flow/flow` folder with `reward-misspecification/flow`. Also add `reward-misspecification/flow_cfg` to `flow/`.

## Changes to original codebase
* Added reward wrapper in ```flow/envs/reward_wrapper.py``` for training proxy rewards. 
* Rewrote training code in ```traffic_train.py```
* Added evaluation code with anomaly detection in ```traffic_eval.py```
* Added proxy rewards in ```flow/core/rewards.py```

The original code is available [here](https://github.com/flow-project/flow).

## Custom experiments
For a custom experiment, train with
```
python3 scripts/traffic_train.py
    --n_cpus $CPUS \ # Using more CPUs will train faster
    --num_steps 25000 \ # Oftentimes the model converges before 25k steps
    --rollout_size 7 \ # At least 7 rollouts are needed for stability
    $EXPERIMENT
    $NAME
    $PROXY
    $PROXY_WEIGHTS
    $WIDTH
    $DEPTH
```
CPUS is the number of CPUS available for training. EXPERIMENT is the experiment type, which should be the name of a configuration file in either `flow_cfg/exp_configs/rl/singleagent` or `flow_cfg/exp_configs/rl/multiagent`. NAME is the name of the experiment. PROXY is a comma-separated list of strings corresponding to the proxy reward. PROXY_WEIGHTS is a comma-separated list of floats corresponding to the weight for each proxy. WIDTH is the number of hidden units in the policy model. DEPTH is the number of layers in the policy model. 

For a custom experiment, evaluate with
```
python3 scripts/traffic_eval.py 
    --baseline $BASELINE_MODEL_PATH \
    $RESULTS_PATH \
    $PROXY \
    $PROXY_WEIGHTS \
    $TRUE \
    $TRUE_WEIGHTS 
```
BASELINE_MODEL_PATH is the path to the model file that will serve as the trusted model. If you are not interested in anomaly detection, skip this. RESULTS_PATH is the path to the folder containing all the model results. PROXY, PROXY_WEIGHTS, TRUE, TRUE_WEIGHTS are similar to the above.

## Reproducing experiments
The learning rate may need to be adjusted depending on the size of the model and the proxies used. In the experiments, the width varied from 4 to 2048 and the layers varied from 1 to 5. 


* For the bottleneck-misweighting experiment, run
```
python3 scripts/traffic_train.py 
    singleagent_bottleneck 
    $NAME 
    desired_vel,forward,lane_bool 
    1,0.1,0.01
    $WIDTH
    $DEPTH
```
and evaluate with
```
python3 scripts/traffic_eval.py 
    $RESULTS_PATH
    desired_vel,forward,lane_bool 
    1,0.1,0.01
    desired_vel,lane_bool 
    1,1
```
* For the merge-misweighting experiment, run
```
python3 scripts/traffic_train.py 
    singleagent_merge
    $NAME 
    vel,accel,headway
    1,0.01,0.1
    $WIDTH
    $DEPTH
```
and evaluate with
```
python3 scripts/traffic_eval.py 
    $RESULTS_PATH
    vel,accel,headway
    1,0.01,0.1
    vel,accel,headway
    1,1,0.1
```
* For the merge-ontological experiment, run
```
python3 scripts/traffic_train.py 
    singleagent_merge_bus 
    $NAME 
    vel,accel,headway 
    1,1,0.1
    $WIDTH
    $DEPTH
```
and evaluate with
```
python3 scripts/traffic_eval.py 
    $RESULTS_PATH
    vel,accel,headway 
    1,1,0.1
    commute,accel,headway
    1,1,0.1
```
* For the merge-scope experiment, run
```
python3 scripts/traffic_train.py 
    singleagent_merge
    $NAME 
    local_first,accel,headway 
    1,1,0.1
    $WIDTH
    $DEPTH
```
and evaluate with
```
python3 scripts/traffic_eval.py 
    $RESULTS_PATH
    local_first,accel,headway 
    1,1,0.1
    vel,accel,headway
    1,1,0.1
```
* For the action-space-resolution experiment, run
```
python3 scripts/traffic_train.py 
    singleagent_merge_bus 
    $EXP_NAME 
    vel,accel,headway,disc_action_noise 
    1,1,0.1,$NOISE
```
and evaluate with 
```
python3 scripts/traffic_eval.py 
    $RESULTS_PATH
    vel,accel,headway,disc_action_noise 
    1,1,0.1,$NOISE
    commute,accel,headway
    1,1,0.1,$NOISE
```

# Flow

[Flow](https://flow-project.github.io/) is a computational framework for deep RL and control experiments for traffic microsimulation.

See [our website](https://flow-project.github.io/) for more information on the application of Flow to several mixed-autonomy traffic scenarios. Other [results and videos](https://sites.google.com/view/ieee-tro-flow/home) are available as well.

# More information

- [Documentation](https://flow.readthedocs.org/en/latest/)
- [Installation instructions](http://flow.readthedocs.io/en/latest/flow_setup.html)
- [Tutorials](https://github.com/flow-project/flow/tree/master/tutorials)
- [Binder Build (beta)](https://mybinder.org/v2/gh/flow-project/flow/binder)

# Technical questions

If you have a bug, please report it. Otherwise, join the [Flow Users group](https://join.slack.com/t/flow-users/shared_invite/enQtODQ0NDYxMTQyNDY2LTY1ZDVjZTljM2U0ODIxNTY5NTQ2MmUxMzYzNzc5NzU4ZTlmNGI2ZjFmNGU4YjVhNzE3NjcwZTBjNzIxYTg5ZmY) on Slack!  

# Getting involved

We welcome your contributions.

- Please report bugs and improvements by submitting [GitHub issue](https://github.com/flow-project/flow/issues).
- Submit your contributions using [pull requests](https://github.com/flow-project/flow/pulls). Please use [this template](https://github.com/flow-project/flow/blob/master/.github/PULL_REQUEST_TEMPLATE.md) for your pull requests.

# Citing Flow

If you use Flow for academic research, you are highly encouraged to cite our paper:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol. abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465

If you use the benchmarks, you are highly encouraged to cite our paper:

Vinitsky, E., Kreidieh, A., Le Flem, L., Kheterpal, N., Jang, K., Wu, F., ... & Bayen, A. M,  Benchmarks for reinforcement learning in mixed-autonomy traffic. In Conference on Robot Learning (pp. 399-409). Available: http://proceedings.mlr.press/v87/vinitsky18a.html

# Contributors

Flow is supported by the [Mobile Sensing Lab](http://bayen.eecs.berkeley.edu/) at UC Berkeley and Amazon AWS Machine Learning research grants. The contributors are listed in [Flow Team Page](https://flow-project.github.io/team.html).
