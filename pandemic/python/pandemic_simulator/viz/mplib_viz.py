# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
import string
from inspect import ismethod
from typing import List, Any, Dict, Optional, Sequence, Type

from datetime import datetime

import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from .evaluation_plots import inf_colors
from .pandemic_viz import PandemicViz
from ..environment import PandemicObservation, InfectionSummary, PandemicSimState, PandemicSimConfig

__all__ = ['BaseMatplotLibViz', 'SimViz', 'GymViz', 'PlotType']


class PlotType:
    global_infection_summary: str = 'gis'
    global_testing_summary: str = 'gts'
    critical_summary: str = 'critical_summary'
    stages: str = 'stages'
    location_assignee_visits: str = 'location_assignee_visits'
    location_visitor_visits: str = 'location_visitor_visits'
    infection_source = 'infection_source'
    cumulative_reward = 'cumulative_reward'
    cumulative_true_reward = 'cumulative_true_reward'

    @staticmethod
    def plot_order() -> List[str]:
        return [PlotType.global_infection_summary, PlotType.global_testing_summary, PlotType.critical_summary,
                PlotType.stages, PlotType.location_assignee_visits, PlotType.location_visitor_visits,
                PlotType.infection_source, PlotType.cumulative_reward, PlotType.cumulative_true_reward]


class BaseMatplotLibViz(PandemicViz):
    """A basic matplotlib visualization for the simulator"""

    _num_persons: int
    _max_hospital_capacity: int
    _axs: List[Axes]
    _ax_i: int

    _gis: List[np.ndarray]
    _gts: List[np.ndarray]
    _stages: List[np.ndarray]
    _rewards: List[float]

    _gis_legend: List[str]
    _critical_index: int
    _stage_indices: np.ndarray

    def __init__(self, num_persons: int, max_hospital_capacity: Optional[int] = None):
        """
        :param num_persons: number of persons in the environment
        :param max_hospital_capacity: maximum hospital capacity, if None, it is set to 1% of the number of persons
        """
        self._num_persons = num_persons
        self._max_hospital_capacity = max_hospital_capacity or min(1, int(0.01 * num_persons))

        self._axs = list()
        self._ax_i = 0

        self._gis = []
        self._gts = []
        self._stages = []

        self._gis_std = []
        self._gts_std = []
        self._stages_std = []

        self._gis_legend = []

        plt.rc('axes', prop_cycle=cycler(color=inf_colors))


    @classmethod
    def from_config(cls: Type['BaseMatplotLibViz'], sim_config: PandemicSimConfig) -> 'BaseMatplotLibViz':
        return cls(num_persons=sim_config.num_persons, max_hospital_capacity=sim_config.max_hospital_capacity)


    def record_obs(self, obs: PandemicObservation) -> None:
        if len(self._gis_legend) == 0:
            self._gis_legend = list(obs.infection_summary_labels)
            self._critical_index = self._gis_legend.index(InfectionSummary.CRITICAL.value)

        self._gis.append(obs.global_infection_summary)
        self._gts.append(obs.global_testing_summary)
        self._stages.append(obs.stage)

    def record_state(self, state: PandemicSimState) -> None:
        obs = PandemicObservation.create_empty()
        obs.update_obs_with_sim_state(state)
        self.record_obs(obs=obs)

    def record(self, data: Any) -> None:
        if isinstance(data, PandemicSimState):
            self.record_state(data)
        elif isinstance(data, PandemicObservation):
            self.record_obs(data)
        else:
            raise ValueError('Unsupported data type')

    def plot_gis(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        ax = ax or plt.gca()
        gis = np.vstack(self._gis).squeeze() * self._num_persons
        ax.plot(gis)
        ax.legend(self._gis_legend, loc=1)
        ax.set_ylim(-0.1, self._num_persons + 1)
        ax.set_title('Global Infection Summary')
        ax.set_xlabel('time (days)')
        ax.set_ylabel('persons')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def plot_gts(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        ax = ax or plt.gca()
        gts = np.vstack(self._gts).squeeze() * self._num_persons
        ax.plot(gts)
        ax.legend(self._gis_legend, loc=1)
        ax.set_ylim(-0.1, self._num_persons + 1)
        ax.set_title('Global Testing Summary')
        ax.set_xlabel('time (days)')
        ax.set_ylabel('persons')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def plot_critical_summary(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        ax = ax or plt.gca()
        gis = np.vstack(self._gis).squeeze() * self._num_persons
        ax.plot(gis[:, self._critical_index])
        ax.plot(np.arange(gis.shape[0]), np.ones(gis.shape[0]) * self._max_hospital_capacity, 'y')
        ax.legend([InfectionSummary.CRITICAL.value, 'Max hospital capacity'], loc=1)
        ax.set_ylim(-0.1, self._max_hospital_capacity * 3)
        ax.set_title('Critical Summary')
        ax.set_xlabel('time (days)')
        ax.set_ylabel('persons')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def plot_stages(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        ax = ax or plt.gca()
        stages = np.concatenate(self._stages).squeeze()
        ax.plot(stages)
        ax.set_ylim(-0.1, kwargs.get('num_stages', np.max(self._stages)) + 1)
        ax.set_title('Stage')
        ax.set_xlabel('time (days)')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    @staticmethod
    def annotate_plot(ax: Axes, label: str) -> None:
        ax.annotate(f'({label})', (0.5, 0.), xytext=(0, -25 - 20),
                    textcoords='offset points', xycoords='axes fraction',
                    ha='center', va='center', size=14)

    def plot(self, name=datetime.now().strftime("%m-%d-%Y-%H_%M_%S"), plots_to_show: Optional[Sequence[str]] = None, *args: Any, **kwargs: Any) -> None:
        if plots_to_show:
            fn_names = [nm for nm in plots_to_show if ismethod(getattr(self, 'plot_' + nm))]
        else:
            fn_names = [nm.split('plot_')[-1] for nm in dir(self) if nm.startswith('plot_') is True]
            fn_names = [nm for nm in sorted(fn_names,
                                            key=lambda x: PlotType.plot_order().index(x)
                                            if x in PlotType.plot_order() else np.inf)]

        plot_fns = [getattr(self, 'plot_' + nm) for nm in fn_names]

        """Make plots"""
        ncols = min(4, len(plot_fns))
        nrows = int(np.ceil(len(plot_fns) / ncols))

        plt.figure(figsize=(4 * ncols, 4 * nrows))

        plot_ref_labels = string.ascii_lowercase
        for ax_i, plot_fn in enumerate(plot_fns):
            ax = plt.subplot(nrows, ncols, ax_i + 1)
            plot_fn(ax, **kwargs)
            self.annotate_plot(ax, plot_ref_labels[ax_i])
        plt.tight_layout()
        if epoch is not None:
            name = str(int(epoch)) + "_" + epoch
        plt.savefig(f"pandemic_policy/{name}")
        plt.show()



class SimViz(BaseMatplotLibViz):
    _loc_assignee_visits: List[np.ndarray]
    _loc_visitor_visits: List[np.ndarray]
    _location_type_to_is: Dict[str, int]
    _loc_types: List[str]
    _person_types: List[str]

    def __init__(self, num_persons: int, max_hospital_capacity: Optional[int] = None):
        """
        :param num_persons: number of persons in the environment
        :param max_hospital_capacity: maximum hospital capacity, if None, it is set to 1% of the number of persons
        """
        super().__init__(num_persons=num_persons, max_hospital_capacity=max_hospital_capacity)
        self._loc_assignee_visits = []
        self._loc_visitor_visits = []
        self._location_type_to_is = {}
        self._loc_types = []
        self._person_types = []

    def record_state(self, state: PandemicSimState) -> None:
        super().record_state(state)

        if len(self._loc_types) == 0:
            self._loc_types = sorted(set(k[0] for k in state.global_location_summary.keys()))
            self._person_types = sorted(set(k[1] for k in state.global_location_summary.keys()))

        _av = np.zeros((1, len(self._loc_types), len(self._person_types)))
        _vv = np.zeros((1, len(self._loc_types), len(self._person_types)))
        for i in range(len(self._loc_types)):
            for j in range(len(self._person_types)):
                ec = state.global_location_summary[(self._loc_types[i], self._person_types[j])].entry_count
                vc = state.global_location_summary[(self._loc_types[i], self._person_types[j])].visitor_count
                _av[0, i, j] = ec - vc
                _vv[0, i, j] = vc
        self._loc_assignee_visits.append(_av)
        self._loc_visitor_visits.append(_vv)
        self._location_type_to_is = {k.__name__: v for k, v in state.location_type_infection_summary.items()}

    def plot_location_assignee_visits(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        ax = ax or plt.gca()
        if len(self._loc_assignee_visits) > 0:
            lv = self._loc_assignee_visits[-1][0]
            x = np.arange(lv.shape[0])
            p = []
            colors = ['g', 'r', 'b']
            bottom = np.zeros(lv.shape[0])
            for j in range(lv.shape[1] - 1, -1, -1):
                p.append(ax.bar(x, lv[:, j], color=colors[j], alpha=0.5, bottom=bottom))
                bottom += lv[:, j]
            ax.set_xticks(x)
            ax.set_xticklabels(self._loc_types, rotation=60, fontsize=8)
            ax.set_title(f'Location Assignee Visits\n(in {len(self._loc_assignee_visits)} days)')
            ax.set_ylabel('num_visits / num_persons')
            ax.set_ylim(0, None)
            ax.legend(p, self._person_types[::-1])

    def plot_location_visitor_visits(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        ax = ax or plt.gca()
        if len(self._loc_visitor_visits) > 0:
            lv = self._loc_visitor_visits[-1][0]
            x = np.arange(lv.shape[0])
            p = []
            colors = ['g', 'r', 'b']
            bottom = np.zeros(lv.shape[0])
            for j in range(lv.shape[1] - 1, -1, -1):
                p.append(ax.bar(x, lv[:, j], color=colors[j], alpha=0.5, bottom=bottom))
                bottom += lv[:, j]
            ax.set_xticks(x)
            ax.set_xticklabels(self._loc_types, rotation=60, fontsize=8)
            ax.set_title(f'Location Visitor Visits\n(in {len(self._loc_visitor_visits)} days)')
            ax.set_ylabel('num_visits / num_persons')
            ax.set_ylim(0, None)
            ax.legend(p, self._person_types[::-1])

    def plot_infection_source(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        ax = ax or plt.gca()
        if len(self._location_type_to_is) > 0:
            x = np.arange(len(self._loc_types))
            ax.bar(x, [self._location_type_to_is[k] / self._num_persons for k in self._loc_types],
                   color='r', alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(self._loc_types, rotation=60, fontsize=8)
            ax.set_ylim(0, None)
            ax.set_title('% Infections / Location Type')
            ax.set_ylabel('% infections')


class GymViz(BaseMatplotLibViz):
    _rewards: List[float]
    _true_rewards: List[float]

    def __init__(self, num_persons: int, max_hospital_capacity: Optional[int] = None):
        """
        :param num_persons: number of persons in the environment
        :param max_hospital_capacity: maximum hospital capacity, if None, it is set to 1% of the number of persons
        """
        super().__init__(num_persons=num_persons, max_hospital_capacity=max_hospital_capacity)
        self._rew = []
        self._rew_std = []
        self._true_rew = []
        self._true_rew_std = []

    def plot_cumulative_reward(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        ax = ax or plt.gca()
        days = np.arange(len(self._rew))
        ax.plot(days, np.cumsum(self._rew))
        #ax.fill_between(days, true_rew - np.array(self._rew_std), true_rew + np.array(self._rew_std), alpha=0.1)
        ax.set_title('Cumulative Reward')
        ax.set_xlabel('time (days)')

    def plot_cumulative_true_reward(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        ax = ax or plt.gca()
        days = np.arange(len(self._true_rew))
        true_rew = np.cumsum(self._true_rew)
        ax.plot(days, true_rew)
        #ax.fill_between(days, true_rew - np.array(self._true_rew_std), true_rew + np.array(self._true_rew_std), alpha=0.1)
        ax.set_title('Cumulative True Reward')
        ax.set_xlabel('time (days)')

    def plot_gis(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        ax = ax or plt.gca()
        gis = np.vstack(self._gis).squeeze()
        gis_std = np.vstack(self._gis_std).squeeze()
        days = np.arange(len(gis))
        for mean, std in zip(np.hsplit(gis, 5), np.hsplit(gis_std, 5)):
            ax.plot(days, mean)
            ax.fill_between(days, (mean-std).squeeze(1), (mean+std).squeeze(1), alpha=0.1)
            ax.legend(self._gis_legend, loc='upper left')
            ax.set_ylim(-0.1, 1.01)
            ax.set_title('Global Infection Summary')
            ax.set_xlabel('time (days)')
            ax.set_ylabel('proportion')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def plot_gts(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        ax = ax or plt.gca()
        gts = np.vstack(self._gts).squeeze()
        gts_std = np.vstack(self._gts_std).squeeze()
        days = np.arange(len(gts))
        for mean, std in zip(np.hsplit(gts, 5), np.hsplit(gts_std, 5)):
            ax.plot(days, mean)
            ax.fill_between(days, (mean-std).squeeze(1), (mean+std).squeeze(1), alpha=0.1)
            ax.legend(self._gis_legend, loc='upper left')
            ax.set_ylim(-0.1, 1.01)
            ax.set_title('Global Testing Summary')
            ax.set_xlabel('time (days)')
            ax.set_ylabel('proportion')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def plot_critical_summary(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        ax = ax or plt.gca()
        gis = np.vstack(self._gis).squeeze()
        gis_std = np.vstack(self._gis_std).squeeze()
        ax.plot(self._num_persons * gis[:, self._critical_index])
        ax.fill_between(np.arange(len(gis)), self._num_persons * (gis-gis_std)[:, self._critical_index], self._num_persons * (gis+gis_std)[:, self._critical_index], alpha=0.1)
        ax.plot(np.arange(gis.shape[0]), np.ones(gis.shape[0]) * self._max_hospital_capacity, 'y')
        ax.legend([InfectionSummary.CRITICAL.value, 'Max hospital capacity'], loc='upper left')
        ax.set_ylim(-0.1, self._max_hospital_capacity * 3)
        ax.set_title('Critical Summary')
        ax.set_xlabel('time (days)')
        ax.set_ylabel('persons')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def plot_stages(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        ax = ax or plt.gca()
        days = np.arange(len(self._stages))
        stages = np.array(self._stages)
        stages_std = np.array(self._stages_std)
        ax.plot(days, stages)
        ax.fill_between(days, stages - stages_std, stages + stages_std, alpha=0.1)
        ax.set_ylim(-0.1, 5) # This assumes at most 5 stages!!
        ax.set_title('Stage')
        ax.set_xlabel('time (days)')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    @staticmethod
    def annotate_plot(ax: Axes, label: str) -> None:
        ax.annotate(f'({label})', (0.5, 0.), xytext=(0, -25 - 20),
                    textcoords='offset points', xycoords='axes fraction',
                    ha='center', va='center', size=14)

    def plot(self, epoch=None, name=datetime.now().strftime("%m-%d-%Y-%H_%M_%S"), plots_to_show: Optional[Sequence[str]] = None, *args: Any, **kwargs: Any) -> None:
        if plots_to_show:
            fn_names = [nm for nm in plots_to_show if ismethod(getattr(self, 'plot_' + nm))]
        else:
            fn_names = [nm.split('plot_')[-1] for nm in dir(self) if nm.startswith('plot_') is True]
            fn_names = [nm for nm in sorted(fn_names,
                                            key=lambda x: PlotType.plot_order().index(x)
                                            if x in PlotType.plot_order() else np.inf)]

        plot_fns = [getattr(self, 'plot_' + nm) for nm in fn_names]

        """Make plots"""
        ncols = min(4, len(plot_fns))
        nrows = int(np.ceil(len(plot_fns) / ncols))

        plt.figure(figsize=(4 * ncols, 4 * nrows))

        plot_ref_labels = string.ascii_lowercase
        for ax_i, plot_fn in enumerate(plot_fns):
            ax = plt.subplot(nrows, ncols, ax_i + 1)
            plot_fn(ax, **kwargs)
            self.annotate_plot(ax, plot_ref_labels[ax_i])
        plt.tight_layout()
        if epoch is not None:
            name = str(int(epoch)) + "_" + name
        plt.savefig(f"pandemic_policy/{name+'.png'}")
        plt.show()

    def record(self, data: Any) -> None:
        if isinstance(data, tuple):
            obs, reward, true_rew = data
            self._rew.append(reward)
            self._true_rew.append(true_rew)
        else:
            obs = data
        assert isinstance(obs, PandemicObservation)
        self.record_obs(obs)

    def record_list(self, obs, gis, gts, stage, rew, true_rew) -> None:
        if len(self._gis_legend) == 0:
            self._gis_legend = list(obs.infection_summary_labels)
            self._critical_index = self._gis_legend.index(InfectionSummary.CRITICAL.value)

        self._gis.append(np.mean(gis, axis=0))
        self._gis_std.append(np.std(gis, axis=0))
        self._gts.append(np.mean(gts, axis=0))
        self._gts_std.append(np.std(gts, axis=0))
        self._stages.append(np.mean(stage))
        self._stages_std.append(np.std(stage))
        self._rew.append(np.mean(rew))
        self._rew_std.append(np.std(rew))
        self._true_rew.append(np.mean(true_rew))
        self._true_rew_std.append(np.std(true_rew))

    def reset(self):
        self._axs = list()
        self._ax_i = 0

        self._gis = []
        self._gts = []
        self._stages = []

        self._gis_std = []
        self._gts_std = []
        self._stages_std = []

        self._rew = []
        self._rew_std = []
        self._true_rew = []
        self._true_rew_std = []

        self._gis_legend = []

        plt.rc('axes', prop_cycle=cycler(color=inf_colors))
