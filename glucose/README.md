# Reward misspecification notes

## Installation
**IMPORTANT: you must use torch==1.3.1; newer versions of torch will not work**

Create a new virtualenv and run ```pip install -r requirements.txt```

## Changes to the original codebase
* Added option to adjust the width and depth of the policy model.
* Added option to adjust the sensor noise from the glucose monitor.
* Added option to calculate the true reward function, which are located in ```bgp/rl/reward_functions.py```
* Added process.py, which extracts the data from the training run and calculates the proxy reward (gylcemic risk) and the true reward (expected cost of treatment).

The original codebase is available [here](https://github.com/MLD3/RL4BG).
### Running experiments
To run experiments, use the following code snippet. Note that the learning rate may need to be adjusted depending on the size of the model.
```
python3 scripts/glucose_train.py $NAME $WIDTH $DEPTH $PROXY $TRUE $NOISE $DEBUG
```
NAME is the experiment name, WIDTH is the hidden size of the policy network, DEPTH is the number of layers in the policy network, PROXY is a string corresponding to the proxy reward function (e.g., 'magni_bg'), TRUE is a string corresponding to the true reward function, NOISE is the magnitude of the sensor noise. Set DEBUG to 'True' for debugging.

In the experiments, the width varied from 64 to 2048 and the depth varied from 1 to 3. The noise was set to 0. The proxy used was 'magni_bg'.

### Evaluation
To evaluate, run process.py. The results will be stored in a .json
```
python3 scripts/glucose_eval.py $RESULTS_PATH $TOP_K
```
RESULTS_PATH is the path of the results and TOP_K specifies the number of episodes (the top K episodes) to collect.

# RL4BG

Public code release for Deep Reinforcement Learning for Closed-Loop Blood Glucose Control, MLHC 2020