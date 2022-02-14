# Reward misspecification notes
## Installation
* Create a new virtualenv and run ```python3 -m pip install -e .```
* Install [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) with ```pip install stable-baselines3```

## Changes to original codebase:
* Added option to include a second wave of infections
* Added option to adjust spotaneous testing rate
* Added political cost proxy, described in the paper
* Added RL training code, built on top of stable-baselines3, in ```pandemic_train.py```
* Added evaluation code with anomaly detection in ```pandemic_eval.py```
* Added wandb logging 

The original code is available [here](https://github.com/SonyAI/PandemicSimulator)

## Running experiments
First, install and setup [wandb](https://docs.wandb.ai/quickstart). The learning rate may need to be adjusted depending on the size of the model and the proxies used. In the experiments, the width varied from 4 to 128 and the layers varied from 1 to 2.

To run the model capacity experiment:
```
python3 scripts/pandemic_train.py 
  --name $NAME \
  --n_cpus $CPUS \
  --width $WIDTH \
  --depth $DEPTH \
  --delta_lo $DELTA_LO \
  --delta_hi $DELTA_HI \
```
To run the observation noise experiment:
```
python3 scripts/pandemic_train.py 
  --name $NAME \
  --n_cpus $CPUS \
  --width $WIDTH \
  --depth $DEPTH \
  --delta_lo $DELTA_LO \
  --delta_hi $DELTA_HI \
  --testing_rate $RATE
``` 
Arguments are detailed in the code. The hyperparameters are documented in [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters). The proxy reward function is given by ```10 * INFECTION_SUMMARY_ABSOLUTE + 0.1 * LOWER_STAGE + 0.01 * SMOOTH_STAGE_CHANGES```. The true reward function is given by ```10 * INFECTION_SUMMARY_ABSOLUTE + 10 * POLITICAL + 0.1 * LOWER_STAGE + 0.01 * SMOOTH_STAGE_CHANGES```.

## Evaluation
Run the ```pandemic_eval.py``` script and ensure that parameters are the same as those used in training. Script is provided as-is and may not work for all use cases.

# PandemicSimulator

This repository implements an agent-based pandemic simulator to analyse and optimize 
mitigation policies that manage economic impact of pandemics without overwhelming hospital capacity. 
Unlike traditional agent-based models, the simulator is able to model fine-grained interactions 
among people at specific locations in a community, at a level of detail that allows the spread of the 
disease to be an emergent property of people’s behaviors and government’s  policies. An interface with 
OpenAI Gym environment is also provided to support an RL-based methodology for optimizing mitigation policies.

The simulator is developed in collaboration between AI researchers and 
epidemiologists and is designed to study pandemics in general. However, 
the current focus is to support the struggle against dealing with the ongoing 
COVID-19 pandemic. 

At a high level, the simulator currently models: 
  - a population with a realistic age distribution, where each person has a 
    distinct pre-defined programmable stochastic (or deterministic) behavior routine 
  - several heterogeneous locations like schools, homes, stores, bars, restaurants, etc., with 
    different characteristics, such as open times, capacity, contact rates, etc. 
  - testing with false positive/negative rates 
  - imperfect public adherence to social distancing measures
  - contact tracing 
  - variable spread rates among infected individuals
    
## Citation
If you plan to use the simulator in your research, please cite us using: 
```
@misc{kompella2020reinforcement,
      title={Reinforcement Learning for Optimization of COVID-19 Mitigation policies}, 
      author={Varun Kompella* and Roberto Capobianco* and Stacy Jong and Jonathan Browne and Spencer Fox and Lauren Meyers and Peter Wurman and Peter Stone},
      year={2020},
      eprint={2010.10560},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Getting Started

### Installation

Clone the git repo, install pip (on Ubuntu, you can just do `apt-get install python3-pip`) and run pip install. 
See `setup.py` for a list of python package requirements that will automatically be installed.
```shell
foo@Foo:~$ git clone https://github.com/SonyAI/PandemicSimulator.git
foo@Foo:~$ cd PandemicSimulator 
foo@Foo:~/PandemicSimulator$ python3 -m pip install -e . 
```

### Quick Example
```python
from tqdm import trange

import pandemic_simulator as ps

# the first thing to do at the start of any experiment is to initialize a few global parameters
ps.init_globals() # you can pass parameters, like a random seed, that are shared across the entire repo.

# init locations
home = ps.env.Home()
work = ps.env.Office()

# init a worker
person = ps.env.Worker(
    person_id=ps.env.PersonID('worker', age=35),  # person_id is a unique id for this person
    home=home.id,  # specify the home_id that person is assigned to
    work=work.id,  # specify the id of the person's workplace
)

# init simulator
sim = ps.env.PandemicSim(locations=[work, home], persons=[person])

# iterate through steps in the simulator, where each step advances an hour
for _ in trange(24, desc='Simulating hour'):
    sim.step()
```

See `scripts/tutorials/` for in-depth tutorials on how to use the simulator.


## Project Structure

`/docs` -- API documentation plus hierarchy. Build it locally by calling `make html` inside docs folder. Requires 
sphinx autodoc package. Installation instructions: https://www.sphinx-doc.org/en/master/usage/installation.html

`/python/pandemic_simulator/` -- Source files

`/scripts` -- Tutorials, calibration-scripts and evaluation scripts

`/test` -- unittests

## Simulator snapshots

![image](https://user-images.githubusercontent.com/6727235/115275002-3f173900-a10f-11eb-9421-b43c17d3ee6c.png)


## Contact
 
 If you have any questions regarding the code, please feel free to contact
 Roberto Capobianco (roberto.capobianco@sony.com) or Varun Kompella (varun.kompella@sony.com).
 
 
###### Copyright 2021, Sony AI, Sony Corporation of America, All rights reserved.
 


