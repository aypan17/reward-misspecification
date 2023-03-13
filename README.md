# Code for Reward Misspecification Experiments
This repository contains code for the paper [The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models](https://arxiv.org/abs/2201.03544).

## Instructions
Each repository has its own installation requirements. We recommend setting up a new virtual environment for each environment and following the instructions provided in each README. The code has been tested using Python 3.7 on machines running Ubuntu 18.04.

## Based off of code from
* flow: https://github.com/flow-project/flow
* PandemicSimulator: https://github.com/SonyAI/PandemicSimulator
* RL4BG: https://github.com/MLD3/RL4BG
* torchbeast: https://github.com/facebookresearch/torchbeast

The `flow`, `pandemic`, `glucose`, `atari` folders hold code for the traffic, COVID, blood glucose monitoring, and atari experiments, respectively. The `flow_cfg` folder holds experiment setup for the traffic experiments.

## Citation
If you use these environments in your own work, please consider citing us!
```
@inproceedings{
    pan2022rewardhacking,
    title={The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models},
    author={Alexander Pan and Kush Bhatia and Jacob Steinhardt},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=JYtwGwIL7ye}
}
```
