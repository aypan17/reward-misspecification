from bgp.simglucose.simulation.scenario import Action, Scenario
import numpy as np
from scipy.stats import truncnorm, uniform
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RandomScenario(Scenario):
    def __init__(self, start_time=None, seed=None, bw=None):
        Scenario.__init__(self, start_time=start_time)
        self.seed = seed

    def get_action(self, t):
        # t must be datetime.datetime object
        delta_t = t - datetime.combine(t.date(), datetime.min.time())
        t_sec = delta_t.total_seconds()

        if t_sec < 1:
            logger.info('Creating new one day scenario ...')
            self.scenario = self.create_scenario()

        t_min = np.floor(t_sec / 60.0)

        if t_min in self.scenario['meal']['time']:
            logger.info('Time for meal!')
            idx = self.scenario['meal']['time'].index(t_min)
            return Action(meal=self.scenario['meal']['amount'][idx])
        else:
            return Action(meal=0)

    def create_scenario(self):
        scenario = {'meal': {'time': [],
                             'amount': []}}

        # Probability of taking each meal
        # [breakfast, snack1, lunch, snack2, dinner, snack3]
        prob = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
        time_lb = np.array([5, 9, 10, 14, 16, 20]) * 60
        time_ub = np.array([9, 10, 14, 16, 20, 23]) * 60
        time_mu = np.array([7, 9.5, 12, 15, 18, 21.5]) * 60
        time_sigma = np.array([60, 30, 60, 30, 60, 30])
        amount_mu = [45, 10, 70, 10, 80, 10]
        amount_sigma = [10, 5, 10, 5, 10, 5]

        for p, tlb, tub, tbar, tsd, mbar, msd in zip(
                prob, time_lb, time_ub, time_mu, time_sigma,
                amount_mu, amount_sigma):
            if self.random_gen.rand() < p:
                tmeal = np.round(truncnorm.rvs(a=(tlb - tbar) / tsd,
                                               b=(tub - tbar) / tsd,
                                               loc=tbar,
                                               scale=tsd,
                                               random_state=self.random_gen))
                scenario['meal']['time'].append(tmeal)
                scenario['meal']['amount'].append(
                    max(round(self.random_gen.normal(mbar, msd)), 0))

        return scenario

    def reset(self):
        self.random_gen = np.random.RandomState(self.seed)
        self.scenario = self.create_scenario()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()


class SemiRandomScenario(Scenario):
    def __init__(self, start_time=None, seed=None, time_std_multiplier=1):
        Scenario.__init__(self, start_time=start_time)
        self.time_std_multiplier = time_std_multiplier
        self.seed = seed

    def get_action(self, t):
        # t must be datetime.datetime object
        delta_t = t - datetime.combine(t.date(), datetime.min.time())
        t_sec = delta_t.total_seconds()

        if t_sec < 1:
            logger.info('Creating new one day scenario ...')
            self.scenario = self.create_scenario()

        t_min = np.floor(t_sec / 60.0)

        if t_min in self.scenario['meal']['time']:
            logger.info('Time for meal!')
            idx = self.scenario['meal']['time'].index(t_min)
            return Action(meal=self.scenario['meal']['amount'][idx])
        else:
            return Action(meal=0)

    def create_scenario(self):
        scenario = {'meal': {'time': [],
                             'amount': []}}

        # Probability of taking each meal
        # [breakfast, snack1, lunch, snack2, dinner, snack3]
        time_lb = np.array([5, 10, 16]) * 60
        time_ub = np.array([10, 16, 22]) * 60
        time_mu = np.array([7.5, 13, 19]) * 60
        time_sigma = np.array([60, 60, 60]) * self.time_std_multiplier
        amount = [45, 70, 80]

        for tlb, tub, tbar, tsd, mbar in zip(time_lb, time_ub, time_mu, time_sigma, amount):
            tmeal = np.round(truncnorm.rvs(a=(tlb - tbar) / tsd,
                                           b=(tub - tbar) / tsd,
                                           loc=tbar,
                                           scale=tsd,
                                           random_state=self.random_gen))
            scenario['meal']['time'].append(tmeal)
            scenario['meal']['amount'].append(mbar)
        return scenario

    def reset(self):
        self.random_gen = np.random.RandomState(self.seed)
        self.scenario = self.create_scenario()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()


def harris_benedict(weight, kind):
    # (min, max, age, weight)
    if kind == 'child':
        child_weight_to_age_and_height = [(0, 25.6, 7, 121.9),
                                          (25.6, 28.6, 8, 128),
                                          (28.6, 32, 9, 133.3),
                                          (32, 35.6, 10, 138.4),
                                          (35.6, 39.9, 11, 143.5),
                                          (39.9, np.infty, 12, 149.1)]
        for tup in child_weight_to_age_and_height:
            if weight > tup[0] and weight <= tup[1]:
                age = tup[2]
                height = tup[3]
    elif kind == 'adolescent':
        adolescent_weight_to_age_and_height = [(0, 50.8, 13, 156.2),
                                               (50.8, 56.0, 14, 163.8),
                                               (56.0, 60.8, 15, 170.1),
                                               (60.8, 64.4, 16, 173.4),
                                               (64.4, 66.9, 17, 175.2),
                                               (66.9, 68.9, 18, 175.7),
                                               (68.9, 70.3, 19, 176.5),
                                               (70.3, np.infty, 20, 177)]
        for tup in adolescent_weight_to_age_and_height:
            if weight > tup[0] and weight <= tup[1]:
                age = tup[2]
                height = tup[3]
    else:
        age = 45
        height = 177
    bmr = 66.5 + (13.75 * weight) + (5.003 * height) - (6.755 * age)
    total = ((1.2 * bmr)*0.45)/4
    adj = 1.1+1.3+1.55+3*.15  # from old carb calc
    b_ratio = 1.1/adj
    l_ratio = 1.3/adj
    d_ratio = 1.55/adj
    s_ratio = .15/adj
    return (total*b_ratio, total*l_ratio, total*d_ratio, total*s_ratio)


class RandomBalancedScenario(Scenario):
    def __init__(self, bw, start_time=None, seed=None, weekly=False,
                 kind=None, harrison_benedict=False, restricted=False, unrealistic=False, meal_duration=1,
                 deterministic_meal_size=False, deterministic_meal_time=False, deterministic_meal_occurrence=False):
        Scenario.__init__(self, start_time=start_time)
        self.kind = kind
        self.bw = bw
        self.day = 0
        self.weekly = weekly
        self.deterministic_meal_size = deterministic_meal_size
        self.deterministic_meal_time = deterministic_meal_time
        self.deterministic_meal_occurrence = deterministic_meal_occurrence
        self.harrison_benedict = harrison_benedict
        self.restricted = restricted  # takes precident over harrison_benedict
        self.unrealistic = unrealistic
        self.meal_duration = meal_duration
        self.seed = seed

    def get_action(self, t):
        # t must be datetime.datetime object
        delta_t = t - datetime.combine(t.date(), datetime.min.time())
        t_sec = delta_t.total_seconds()

        if t_sec < 1:
            logger.info('Creating new one day scenario ...')
            self.day = (self.day + 1) % 7
            if self.restricted:
                self.scenario = self.create_scenario_restricted()
            elif self.harrison_benedict:
                self.scenario = self.create_scenario_harrison_benedict()
            elif self.unrealistic:
                self.scenario = self.create_scenario_unrealistic()
            elif self.weekly:
                if self.day == 5 or self.day == 6:
                    self.scenario = self.create_weekend_scenario()
                else:
                    self.scenario = self.create_scenario()
            else:
                self.scenario = self.create_scenario()

        t_min = np.floor(t_sec / 60.0)

        # going to cancel overlapping meals by going with first meal in range
        for idx, time in enumerate(self.scenario['meal']['time']):
            if t_min>=time and t_min<time+self.meal_duration:
                return Action(meal=self.scenario['meal']['amount'][idx]/self.meal_duration)
        else:
            return Action(meal=0)

    def create_scenario_restricted(self):
        scenario = {'meal': {'time': [],
                             'amount': []}}
        # Probability of taking each meal
        # [breakfast, snack1, lunch, snack2, dinner, snack3]
        prob = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
        time_lb = np.array([5, 9, 10, 14, 16, 20]) * 60
        time_ub = np.array([9, 10, 14, 16, 20, 23]) * 60
        time_mu = np.array([7, 9.5, 12, 15, 18, 21.5]) * 60
        time_sigma = np.array([60, 30, 60, 30, 60, 30])
        if self.kind == 'child':
            amount_lb = np.array([30, 10, 30, 10, 30, 10])
            amount_ub = np.array([45, 20, 45, 20, 45, 20])
        elif self.kind == 'adolescent':
            amount_lb = np.array([45, 15, 45, 15, 45, 15])
            amount_ub = np.array([60, 25, 60, 25, 60, 25])
        elif self.kind == 'adult':
            amount_lb = np.array([60, 20, 60, 20, 60, 20])
            amount_ub = np.array([75, 30, 75, 30, 75, 30])
        else:
            raise ValueError('{} not a valid kind (child, adolescent, adult)'.format(self.kind))
        amount_mu = np.array([0.7, 0.15, 1.1, 0.15, 1.25, 0.15]) * self.bw
        amount_sigma = amount_mu * 0.15
        for p, tlb, tub, tbar, tsd, mlb, mub, mbar, msd in zip(
                prob, time_lb, time_ub, time_mu, time_sigma,
                amount_lb, amount_ub, amount_mu, amount_sigma):
            if self.random_gen.rand() < p:
                tmeal = np.round(truncnorm.rvs(a=(tlb - tbar) / tsd,
                                               b=(tub - tbar) / tsd,
                                               loc=tbar,
                                               scale=tsd,
                                               random_state=self.random_gen))
                scenario['meal']['time'].append(tmeal)
                ameal = np.round(truncnorm.rvs(a=(mlb - mbar) / msd,
                                               b=(mub - mbar) / msd,
                                               loc=mbar,
                                               scale=msd,
                                               random_state=self.random_gen))
                scenario['meal']['amount'].append(ameal)
        return scenario

    def create_scenario_unrealistic(self):
        scenario = {'meal': {'time': [],
                             'amount': []}}
        # Probability of taking each meal
        # [breakfast, snack1, lunch, snack2, dinner, snack3]
        prob = [1, 1, 1]

        time_lb = np.array([8, 12, 18]) * 60
        time_ub = np.array([9, 15, 19]) * 60
        time_mu = np.array([9, 14, 19]) * 60
        time_sigma = np.array([0, 0, 0])
        amount_mu = np.array([50, 80, 60])
        amount_sigma = np.array([0, 0, 0])
        for p, tlb, tub, tbar, tsd, mbar, msd in zip(
                prob, time_lb, time_ub, time_mu, time_sigma,
                amount_mu, amount_sigma):
            if self.random_gen.rand() < p:
                tmeal = np.round(truncnorm.rvs(a=-np.infty,
                                               b=np.infty,
                                               loc=tbar,
                                               scale=tsd,
                                               random_state=self.random_gen))
                scenario['meal']['time'].append(tmeal)
                ameal = np.round(truncnorm.rvs(a=-np.infty,
                                               b=np.infty,
                                               loc=mbar,
                                               scale=msd,
                                               random_state=self.random_gen))
                scenario['meal']['amount'].append(ameal)
        return scenario

    def create_scenario_harrison_benedict(self):
        scenario = {'meal': {'time': [],
                             'amount': []}}
        # Probability of taking each meal
        # [breakfast, snack1, lunch, snack2, dinner, snack3]
        mu_b, mu_l, mu_d, mu_s = harris_benedict(self.bw, self.kind)
        prob = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
        time_lb = np.array([5, 9, 10, 14, 16, 20]) * 60
        time_ub = np.array([9, 10, 14, 16, 20, 23]) * 60
        time_mu = np.array([7, 9.5, 12, 15, 18, 21.5]) * 60
        time_sigma = np.array([60, 30, 60, 30, 60, 30])
        amount_mu = np.array([mu_b, mu_s, mu_l, mu_s, mu_d, mu_s])
        amount_sigma = amount_mu * 0.15
        # TODO: left off adding in a truncnorm option independent of bw if I feed in patient kind
        for p, tlb, tub, tbar, tsd, mbar, msd in zip(
                prob, time_lb, time_ub, time_mu, time_sigma, amount_mu, amount_sigma):
            if self.random_gen.rand() < p:
                tmeal = np.round(truncnorm.rvs(a=(tlb - tbar) / tsd,
                                               b=(tub - tbar) / tsd,
                                               loc=tbar,
                                               scale=tsd,
                                               random_state=self.random_gen))
                scenario['meal']['time'].append(tmeal)
                ameal = np.round(self.random_gen.normal(mbar, msd))
                scenario['meal']['amount'].append(ameal)
        return scenario

    def create_scenario(self):  # TODO
        scenario = {'meal': {'time': [],
                             'amount': []}}

        # Probability of taking each meal
        # [breakfast, snack1, lunch, snack2, dinner, snack3]
        prob = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
        time_lb = np.array([5, 9, 10, 14, 16, 20]) * 60
        time_ub = np.array([9, 10, 14, 16, 20, 23]) * 60
        time_mu = np.array([7, 9.5, 12, 15, 18, 21.5]) * 60
        time_sigma = np.array([60, 30, 60, 30, 60, 30])
        #if self.kind == 'child':
        #    pass
        #elif self.kind == 'adolescent':
        #    pass
        #elif self.kind == 'adult':
        #    pass
        #else:
        amount_mu = np.array([0.7, 0.15, 1.1, 0.15, 1.25, 0.15]) * self.bw
        amount_sigma = amount_mu * 0.15
        # TODO: left off adding in a truncnorm option independent of bw if I feed in patient kind
        for p, tlb, tub, tbar, tsd, mbar, msd in zip(
                prob, time_lb, time_ub, time_mu, time_sigma,
                amount_mu, amount_sigma):
            if self.random_gen.rand() < p or self.deterministic_meal_occurrence:
                tmeal = np.round(truncnorm.rvs(a=(tlb - tbar) / tsd,
                                               b=(tub - tbar) / tsd,
                                               loc=tbar,
                                               scale=tsd,
                                               random_state=self.random_gen))
                ameal = max(round(self.random_gen.normal(mbar, msd)), 0)
                if self.deterministic_meal_time:
                    tmeal = np.round(tbar)
                if self.deterministic_meal_size:
                    ameal = round(mbar, 0)
                scenario['meal']['time'].append(tmeal)
                scenario['meal']['amount'].append(ameal)

        return scenario

    def create_weekend_scenario(self):
        scenario = {'meal': {'time': [],
                             'amount': []}}

        # Probability of taking each meal
        # [breakfast, snack1, lunch, snack2, dinner, snack3]
        prob = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
        time_lb = np.array([7, 11, 12, 15, 18, 22]) * 60
        time_ub = np.array([11, 12, 15, 16, 22, 23]) * 60
        time_mu = np.array([9, 11.5, 13.5, 15.5, 21, 22.5]) * 60
        time_sigma = np.array([60, 30, 60, 30, 60, 30])
        amount_mu = np.array([1.1, 0.15, 1.3, 0.15, 1.55, 0.15]) * self.bw
        amount_sigma = amount_mu * 0.15

        for p, tlb, tub, tbar, tsd, mbar, msd in zip(
                prob, time_lb, time_ub, time_mu, time_sigma,
                amount_mu, amount_sigma):
            if self.random_gen.rand() < p:
                tmeal = np.round(truncnorm.rvs(a=(tlb - tbar) / tsd,
                                               b=(tub - tbar) / tsd,
                                               loc=tbar,
                                               scale=tsd,
                                               random_state=self.random_gen))
                scenario['meal']['time'].append(tmeal)
                scenario['meal']['amount'].append(
                    max(round(self.random_gen.normal(mbar, msd)), 0))

        return scenario

    def reset(self):
        self.random_gen = np.random.RandomState(self.seed)
        self.scenario = self.create_scenario()
        self.day = 0

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()


class CustomBalancedScenario(Scenario):
    # for right now, assuming meals uniform at random in time partitioned in the day
    def __init__(self, bw, start_time=None, seed=None, num_meals=3, size_mult=1):
        Scenario.__init__(self, start_time=start_time)
        self.bw = bw
        self.num_meals = num_meals
        self.size_mult = size_mult
        self.seed = seed

    def get_action(self, t):
        # t must be datetime.datetime object
        delta_t = t - datetime.combine(t.date(), datetime.min.time())
        t_sec = delta_t.total_seconds()

        if t_sec < 1:
            logger.info('Creating new one day scenario ...')
            self.scenario = self.create_scenario()

        t_min = np.floor(t_sec / 60.0)

        if t_min in self.scenario['meal']['time']:
            logger.info('Time for meal!')
            idx = self.scenario['meal']['time'].index(t_min)
            return Action(meal=self.scenario['meal']['amount'][idx])
        else:
            return Action(meal=0)

    def create_scenario(self):
        scenario = {'meal': {'time': [], 'amount': []}}
        daily_mins = 1440
        time_bins = [((daily_mins/self.num_meals)*i, (daily_mins/self.num_meals)*(i+1)) for i in range(self.num_meals)]
        time_lb = np.array([tb[0] for tb in time_bins])
        time_ub = np.array([tb[1] for tb in time_bins])
        amount_per_meal = min(1.25*self.size_mult, (3*self.size_mult)/self.num_meals)
        amount_mu = np.array([amount_per_meal for _ in range(self.num_meals)]) * self.bw
        amount_sigma = amount_mu * 0.15
        for tlb, tub, mbar, msd in zip(time_lb, time_ub, amount_mu, amount_sigma):
            tmeal = np.round(uniform.rvs(loc=tlb, scale=tub-tlb, random_state=self.random_gen))
            ameal = max(round(self.random_gen.normal(mbar, msd)), 0)
            scenario['meal']['time'].append(tmeal)
            scenario['meal']['amount'].append(ameal)
        return scenario

    def reset(self):
        self.random_gen = np.random.RandomState(self.seed)
        self.scenario = self.create_scenario()
        self.day = 0

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()


class SemiRandomBalancedScenario(Scenario):
    def __init__(self, bw, start_time=None, seed=None, time_std_multiplier=1,
                 kind=None, harrison_benedict=False, meal_duration=1):
        Scenario.__init__(self, start_time=start_time)
        self.time_std_multiplier = time_std_multiplier
        self.bw = bw
        self.kind = kind
        self.harrison_benedict = harrison_benedict
        self.meal_duration = meal_duration
        self.seed = seed

    def get_action(self, t):
        # t must be datetime.datetime object
        delta_t = t - datetime.combine(t.date(), datetime.min.time())
        t_sec = delta_t.total_seconds()

        if t_sec < 1:
            logger.info('Creating new one day scenario ...')
            if self.harrison_benedict:
                self.scenario = self.create_scenario_harrison_benedict()
            else:
                self.scenario = self.create_scenario()

        t_min = np.floor(t_sec / 60.0)

        # going to cancel overlapping meals by going with first meal in range
        for idx, time in enumerate(self.scenario['meal']['time']):
            if t_min >= time and t_min < time + self.meal_duration:
                return Action(meal=self.scenario['meal']['amount'][idx] / self.meal_duration)
        else:
            return Action(meal=0)

    def create_scenario(self):
        scenario = {'meal': {'time': [],
                             'amount': []}}

        # Probability of taking each meal
        # [breakfast, snack1, lunch, snack2, dinner, snack3]
        time_lb = np.array([5, 10, 16]) * 60
        time_ub = np.array([10, 16, 22]) * 60
        time_mu = np.array([7.5, 13, 19]) * 60
        time_sigma = np.array([60, 60, 60]) * self.time_std_multiplier
        amount = np.array([0.7, 1.1, 1.25]) * self.bw

        for tlb, tub, tbar, tsd, mbar in zip(time_lb, time_ub, time_mu, time_sigma, amount):
            tmeal = np.round(truncnorm.rvs(a=(tlb - tbar) / tsd,
                                           b=(tub - tbar) / tsd,
                                           loc=tbar,
                                           scale=tsd,
                                           random_state=self.random_gen))
            scenario['meal']['time'].append(tmeal)
            scenario['meal']['amount'].append(mbar)
        return scenario

    def create_scenario_harrison_benedict(self):
        scenario = {'meal': {'time': [],
                             'amount': []}}
        # Probability of taking each meal
        # [breakfast, snack1, lunch, snack2, dinner, snack3]
        mu_b, mu_l, mu_d, mu_s = harris_benedict(self.bw, self.kind)
        time_lb = np.array([5, 10, 16]) * 60
        time_ub = np.array([10, 16, 22]) * 60
        time_mu = np.array([7.5, 13, 19]) * 60
        time_sigma = np.array([60, 60, 60]) * self.time_std_multiplier
        amount_mu = np.array([mu_b, mu_l, mu_d])
        for tlb, tub, tbar, tsd, mbar in zip(
                time_lb, time_ub, time_mu, time_sigma, amount_mu):
            tmeal = np.round(truncnorm.rvs(a=(tlb - tbar) / tsd,
                                           b=(tub - tbar) / tsd,
                                           loc=tbar,
                                           scale=tsd,
                                           random_state=self.random_gen))
            scenario['meal']['time'].append(tmeal)
            ameal = np.round(mbar)
            scenario['meal']['amount'].append(ameal)
        return scenario

    def reset(self):
        self.random_gen = np.random.RandomState(self.seed)
        self.scenario = self.create_scenario()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()


if __name__ == '__main__':
    from datetime import time
    from datetime import timedelta
    import copy
    now = datetime.now()
    t0 = datetime.combine(now.date(), time(6, 0, 0, 0))
    t = copy.deepcopy(t0)
    sim_time = timedelta(days=2)

    scenario = RandomScenario(seed=1)
    m = []
    T = []
    while t < t0 + sim_time:
        action = scenario.get_action(t)
        m.append(action.meal)
        T.append(t)
        t += timedelta(minutes=1)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.plot(T, m)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(mdates.AutoDateLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    plt.show()
