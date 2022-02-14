"""A series of reward functions."""

import numpy as np

def local_desired_velocity(env, veh_ids, fail=False):
    """ 
    Encourage proximity to a desired velocity. 
    We only observe the velocity of the specified car.
    If a collison or failure occurs, we return 0.
    """
    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    num_vehicles = len(veh_ids)

    if any(vel < -100) or fail or num_vehicles == 0:
        return 0.

    target_vel = env.env_params.additional_params['target_velocity']
    max_cost = np.array([target_vel] * num_vehicles)
    max_cost = np.linalg.norm(max_cost)

    cost = vel - target_vel
    cost = np.linalg.norm(cost)

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    return max(max_cost - cost, 0) / (max_cost + eps)


def desired_velocity(env, fail=False, edge_list=None):
    r"""Encourage proximity to a desired velocity.

    This function measures the deviation of a system of vehicles from a
    user-specified desired velocity peaking when all vehicles in the ring
    are set to this desired velocity. Moreover, in order to ensure that the
    reward function naturally punishing the early termination of rollouts due
    to collisions or other failures, the function is formulated as a mapping
    :math:`r: \\mathcal{S} \\times \\mathcal{A}
    \\rightarrow \\mathbb{R}_{\\geq 0}`.
    This is done by subtracting the deviation of the system from the
    desired velocity from the peak allowable deviation from the desired
    velocity. Additionally, since the velocity of vehicles are
    unbounded above, the reward is bounded below by zero,
    to ensure nonnegativity.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    fail : bool, optional
        specifies if any crash or other failure occurred in the system
    edge_list : list  of str, optional
        list of edges the reward is computed over. If no edge_list is defined,
        the reward is computed over all edges

    Returns
    -------
    float
        reward value
    """
    if edge_list is None:
        veh_ids = env.k.vehicle.get_ids()
    else:
        veh_ids = env.k.vehicle.get_ids_by_edge(edge_list)

    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    num_vehicles = len(veh_ids)

    if any(vel < -100) or fail or num_vehicles == 0:
        return 0.

    target_vel = env.env_params.additional_params['target_velocity']
    max_cost = np.array([target_vel] * num_vehicles)
    max_cost = np.linalg.norm(max_cost)

    cost = vel - target_vel
    cost = np.linalg.norm(cost)

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    return max(max_cost - cost, 0) / (max_cost + eps)


def average_velocity(env, fail=False):
    """Encourage proximity to an average velocity.

    This reward function returns the average velocity of all
    vehicles in the system.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    fail : bool, optional
        specifies if any crash or other failure occurred in the system

    Returns
    -------
    float
        reward value
    """
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

    if any(vel < -100) or fail:
        return 0.
    if len(vel) == 0:
        return 0.

    return np.mean(vel)


def rl_forward_progress(env, gain=0.1):
    """Rewared function used to reward the RL vehicles for travelling forward.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    gain : float
        specifies how much to reward the RL vehicles

    Returns
    -------
    float
        reward value
    """
    rl_velocity = env.k.vehicle.get_speed(env.k.vehicle.get_rl_ids())
    rl_norm_vel = np.linalg.norm(rl_velocity, 1)
    return rl_norm_vel * gain


def boolean_action_penalty(discrete_actions, gain=1.0):
    """Penalize boolean actions that indicate a switch."""
    return gain * np.sum(discrete_actions)


def min_delay(env):
    """Reward function used to encourage minimization of total delay.

    This function measures the deviation of a system of vehicles from all the
    vehicles smoothly travelling at a fixed speed to their destinations.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.

    Returns
    -------
    float
        reward value
    """
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

    vel = vel[vel >= -1e-6]
    v_top = max(
        env.k.network.speed_limit(edge)
        for edge in env.k.network.get_edge_list())
    time_step = env.sim_step

    max_cost = time_step * sum(vel.shape)

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    cost = time_step * sum((v_top - vel) / v_top)
    return max((max_cost - cost) / (max_cost + eps), 0)


def avg_delay_specified_vehicles(env, veh_ids):
    """Calculate the average delay for a set of vehicles in the system.

    Parameters
    ----------
    env: flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    veh_ids: a list of the ids of the vehicles, for which we are calculating
        average delay
    Returns
    -------
    float
        average delay
    """
    sum = 0
    for edge in env.k.network.get_edge_list():
        for veh_id in env.k.vehicle.get_ids_by_edge(edge):
            v_top = env.k.network.speed_limit(edge)
            sum += (v_top - env.k.vehicle.get_speed(veh_id)) / v_top
    time_step = env.sim_step
    try:
        cost = time_step * sum
        return cost / len(veh_ids)
    except ZeroDivisionError:
        return 0


def min_delay_unscaled(env):
    """Return the average delay for all vehicles in the system.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.

    Returns
    -------
    float
        reward value
    """
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

    vel = vel[vel >= -1e-6]
    v_top = max(
        env.k.network.speed_limit(edge)
        for edge in env.k.network.get_edge_list())
    time_step = env.sim_step

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    cost = time_step * sum((v_top - vel) / v_top)
    return cost / (env.k.vehicle.num_vehicles + eps)


def penalize_standstill(env, gain=1):
    """Reward function that penalizes vehicle standstill.

    Is it better for this to be:
        a) penalize standstill in general?
        b) multiplicative based on time that vel=0?

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    gain : float
        multiplicative factor on the action penalty

    Returns
    -------
    float
        reward value
    """
    veh_ids = env.k.vehicle.get_ids()
    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    num_standstill = len(vel[vel == 0])
    penalty = gain * num_standstill
    return -penalty


def penalize_near_standstill(env, thresh=0.3, gain=1):
    """Reward function which penalizes vehicles at a low velocity.

    This reward function is used to penalize vehicles below a
    specified threshold. This assists with discouraging RL from
    gamifying a network, which can result in standstill behavior
    or similarly bad, near-zero velocities.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
    thresh : float
        the velocity threshold below which penalties are applied
    gain : float
        multiplicative factor on the action penalty
    """
    veh_ids = env.k.vehicle.get_ids()
    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    penalize = len(vel[vel < thresh])
    penalty = gain * penalize
    return -penalty


def penalize_headway_variance(vehicles,
                              vids,
                              normalization=1,
                              penalty_gain=1,
                              penalty_exponent=1):
    """Reward function used to train rl vehicles to encourage large headways.

    Parameters
    ----------
    vehicles : flow.core.kernel.vehicle.KernelVehicle
        contains the state of all vehicles in the network (generally
        self.vehicles)
    vids : list of str
        list of ids for vehicles
    normalization : float, optional
        constant for scaling (down) the headways
    penalty_gain : float, optional
        sets the penalty for each vehicle between 0 and this value
    penalty_exponent : float, optional
        used to allow exponential punishing of smaller headways
    """
    headways = penalty_gain * np.power(
        np.array(
            [vehicles.get_headway(veh_id) / normalization
             for veh_id in vids]), penalty_exponent)
    return -np.var(headways)


def punish_rl_lane_changes(env, penalty=1):
    """Penalize an RL vehicle performing lane changes.

    This reward function is meant to minimize the number of lane changes and RL
    vehicle performs.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    penalty : float, optional
        penalty imposed on the reward function for any rl lane change action
    """
    total_lane_change_penalty = 0
    for veh_id in env.k.vehicle.get_rl_ids():
        if env.k.vehicle.get_last_lc(veh_id) == env.timer:
            total_lane_change_penalty -= penalty

    return total_lane_change_penalty


def energy_consumption(env, gain=.001):
    """Calculate power consumption of a vehicle.

    Assumes vehicle is an average sized vehicle.
    The power calculated here is the lower bound of the actual power consumed
    by a vehicle.
    """
    power = 0

    M = 1200  # mass of average sized vehicle (kg)
    g = 9.81  # gravitational acceleration (m/s^2)
    Cr = 0.005    # rolling resistance coefficient
    Ca = 0.3  # aerodynamic drag coefficient
    rho = 1.225  # air density (kg/m^3)
    A = 2.6  # vehicle cross sectional area (m^2)
    for veh_id in env.k.vehicle.get_ids():
        speed = env.k.vehicle.get_speed(veh_id)
        prev_speed = env.k.vehicle.get_previous_speed(veh_id)

        accel = abs(speed - prev_speed) / env.sim_step

        power += M * speed * accel + M * g * Cr * speed + 0.5 * rho * A * Ca * speed ** 3

    return -gain * power


def veh_energy_consumption(env, veh_id, gain=.001):
    """Calculate power consumption of a vehicle.

    Assumes vehicle is an average sized vehicle.
    The power calculated here is the lower bound of the actual power consumed
    by a vehicle.
    """
    power = 0

    M = 1200  # mass of average sized vehicle (kg)
    g = 9.81  # gravitational acceleration (m/s^2)
    Cr = 0.005    # rolling resistance coefficient
    Ca = 0.3  # aerodynamic drag coefficient
    rho = 1.225  # air density (kg/m^3)
    A = 2.6  # vehicle cross sectional area (m^2)
    speed = env.k.vehicle.get_speed(veh_id)
    prev_speed = env.k.vehicle.get_previous_speed(veh_id)

    accel = abs(speed - prev_speed) / env.sim_step

    power += M * speed * accel + M * g * Cr * speed + 0.5 * rho * A * Ca * speed ** 3

    return -gain * power


def miles_per_megajoule(env, veh_ids=None, gain=.001):
    """Calculate miles per mega-joule of either a particular vehicle or the total average of all the vehicles.

    Assumes vehicle is an average sized vehicle.
    The power calculated here is the lower bound of the actual power consumed
    by a vehicle.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    veh_ids : [list]
        list of veh_ids to compute the reward over
    gain : float
        scaling factor for the reward
    """
    mpj = 0
    counter = 0
    if veh_ids is None:
        veh_ids = env.k.vehicle.get_ids()
    elif not isinstance(veh_ids, list):
        veh_ids = [veh_ids]
    for veh_id in veh_ids:
        speed = env.k.vehicle.get_speed(veh_id)
        # convert to be positive since the function called is a penalty
        power = -veh_energy_consumption(env, veh_id, gain=1.0)
        if power > 0 and speed >= 0.0:
            counter += 1
            # meters / joule is (v * \delta t) / (power * \delta t)
            mpj += speed / power
    if counter > 0:
        mpj /= counter

    # convert from meters per joule to miles per joule
    mpj /= 1609.0
    # convert from miles per joule to miles per megajoule
    mpj *= 10**6

    return mpj * gain


def miles_per_gallon(env, veh_ids=None, gain=.001):
    """Calculate mpg of either a particular vehicle or the total average of all the vehicles.

    Assumes vehicle is an average sized vehicle.
    The power calculated here is the lower bound of the actual power consumed
    by a vehicle.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    veh_ids : [list]
        list of veh_ids to compute the reward over
    gain : float
        scaling factor for the reward
    """
    mpg = 0
    counter = 0
    if veh_ids is None:
        veh_ids = env.k.vehicle.get_ids()
    elif not isinstance(veh_ids, list):
        veh_ids = [veh_ids]
    for veh_id in veh_ids:
        speed = env.k.vehicle.get_speed(veh_id)
        gallons_per_s = env.k.vehicle.get_fuel_consumption(veh_id)
        if gallons_per_s > 0 and speed >= 0.0:
            counter += 1
            # meters / gallon is (v * \delta t) / (gallons_per_s * \delta t)
            mpg += speed / gallons_per_s
    if counter > 0:
        mpg /= counter

    # convert from meters per gallon to miles per gallon
    mpg /= 1609.0

    return mpg * gain

###################################### REWARD REGISTRY ######################################
def upweight_bus_vel(env, rl_actions):
    bus_vel = np.array([env.k.vehicle.get_speed(veh_id) for veh_id in env.k.vehicle.get_ids() if veh_id[:3] == 'bus'])
    other_vel = np.array([env.k.vehicle.get_speed(veh_id) for veh_id in env.k.vehicle.get_ids() if veh_id[:3] != 'bus'])
    bus = np.mean(bus_vel) if bus_vel.size != 0 else 0 
    other = np.mean(other_vel) if other_vel.size != 0 else 0
    return 60 * bus + other

def global_vel_all(env, rl_actions):
    vel = np.array([
        env.k.vehicle.get_speed(veh_id)
        for veh_id in env.k.vehicle.get_ids()
    ])
    return np.mean(vel) if vel.size != 0 else 0 

def commute_time(env, rl_actions):
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))
  
    if any(vel < -100):
        return -10000.
    if len(vel) == 0:
        return -10000.

    commute = np.array([(v+0.001) ** -1 for v in vel])
    commute = commute[commute > 0]
    return -np.mean(commute)

def desired_velocity_all(env, rl_actions):
    return desired_velocity(env)

def local_desired_vel_first(env, rl_actions):
    return local_desired_velocity(env, env.rl_veh[:2])

def local_desired_vel_last(env, rl_actions):
    return local_desired_velocity(env, env.rl_veh[-2:])

def local_desired_vel_all(env, rl_actions):
    return local_desired_velocity(env, env.rl_veh)

def outflow(env, rl_actions):
    return env.k.vehicle.get_outflow_rate(10 * env.sim_step)

def forward_progress(env, rl_actions):
    rl_vel = env.k.vehicle.get_speed(env.k.vehicle.get_rl_ids())
    return np.linalg.norm(rl_vel, 1)

def penalize_mean_delay(env, rl_actions):
  return -min_delay_unscaled(env)

def penalize_still(env, rl_actions):
  return penalize_standstill(env)

def penalize_headway(env, rl_actions):
    cost = 0
    t_min = 1  # smallest acceptable time headway
    for rl_id in env.rl_veh:
        lead_id = env.k.vehicle.get_leader(rl_id)
        if lead_id not in ["", None] and env.k.vehicle.get_speed(rl_id) > 0:
            t_headway = max(env.k.vehicle.get_headway(rl_id) / env.k.vehicle.get_speed(rl_id), 0)
            cost += min((t_headway - t_min) / t_min, 0)
    return cost 

def penalize_accel(env, rl_actions):
    if rl_actions is None:
        return 0
    mean_actions = np.mean(np.abs(np.array(rl_actions)))
    accel_threshold = 0
    return min(0, accel_threshold - mean_actions)

def penalize_lane_change(env, rl_actions):
    reward = 0
    for veh_id in env.k.vehicle.get_rl_ids():
        if env.k.vehicle.get_last_lc(veh_id) == env.time_counter:
            reward -= 0.1
    return reward

def penalize_boolean_lane_change(env, rl_actions):
    if rl_actions is None:
        return 0
    lane_change_acts = np.abs(np.round(rl_actions[1::2])[:env.k.vehicle.num_rl_vehicles])
    return -np.sum(lane_change_acts)

def penalize_light_change(env, rl_actions):
    if rl_actions is None:
        return 0
    return -np.sum(rl_actions >= 0.5)

def penalize_cars(env, rl_actions):
    return -len(env.k.vehicle.get_ids())

REWARD_REGISTRY = {
  "bus_vel": upweight_bus_vel,
  "bus": upweight_bus_vel,
  "commute": commute_time,
  "time": commute_time,
  "vel": global_vel_all,
  "velocity": global_vel_all,
  "desired_vel": desired_velocity_all,
  "desired_velocity": desired_velocity_all,
  "local_first": local_desired_vel_first,
  "partial_first": local_desired_vel_first,
  "local_last": local_desired_vel_last,
  "partial_last": local_desired_vel_last,
  "local": local_desired_vel_all,
  "partial_all": local_desired_vel_all,
  "outflow": outflow,
  "out": outflow,
  "forward_progress": forward_progress,
  "forward": forward_progress,
  "min_delay": penalize_mean_delay,
  "delay": penalize_mean_delay,
  "headway": penalize_headway,
  "accel": penalize_accel,
  "lane_change": penalize_lane_change,
  "lane": penalize_lane_change,
  "lane_change_bool": penalize_boolean_lane_change,
  "lane_bool": penalize_boolean_lane_change,
  "light_change": penalize_light_change,
  "light": penalize_light_change,
  "cars": penalize_cars,
  "standstill": penalize_still,
  "still": penalize_still,
}
