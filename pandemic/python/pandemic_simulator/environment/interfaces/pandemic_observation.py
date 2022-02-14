# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from dataclasses import dataclass
from typing import Sequence, Type, cast, Optional, Sequence

import numpy as np

from .ids import LocationID
from .infection_model import InfectionSummary, sorted_infection_summary
from .location_states import NonEssentialBusinessLocationState
from .sim_state import PandemicSimState

__all__ = ['PandemicObservation']


@dataclass
class PandemicObservation:
    """Dataclass that updates numpy arrays with information from PandemicSimState. Typically, this observation is
    used by the reinforcement learning interface."""

    global_infection_summary: np.ndarray
    global_infection_summary_alpha: np.ndarray
    global_infection_summary_delta: np.ndarray
    global_testing_summary: np.ndarray
    global_testing_summary_alpha: np.ndarray
    global_testing_summary_delta: np.ndarray
    stage: np.ndarray
    infection_above_threshold: np.ndarray
    time_day: np.ndarray
    state: PandemicSimState
    unlocked_non_essential_business_locations: Optional[np.ndarray] = None
    
    @classmethod
    def create_empty(cls: Type['PandemicObservation'],
                     history_size: int = 1,
                     num_non_essential_business: Optional[int] = None) -> 'PandemicObservation':
        """
        Creates an empty observation TNC layout array.

        :param history_size: Size of history. If set > 1, the observation can hold information from multiple sequences
            of PandemicSimStates.
        :param num_non_essential_business: Number of non essential business locations.
        :return: an empty PandemicObservation instance
        """
        return PandemicObservation(global_infection_summary=np.zeros((history_size, 1, len(InfectionSummary))),
                                   global_infection_summary_alpha=np.zeros((history_size, 1, len(InfectionSummary))),
                                   global_infection_summary_delta=np.zeros((history_size, 1, len(InfectionSummary))),
                                   global_testing_summary=np.zeros((history_size, 1, len(InfectionSummary))),
                                   global_testing_summary_alpha=np.zeros((history_size, 1, len(InfectionSummary))),
                                   global_testing_summary_delta=np.zeros((history_size, 1, len(InfectionSummary))),
                                   stage=np.zeros((history_size, 1, 1)),
                                   infection_above_threshold=np.zeros((history_size, 1, 1)),
                                   time_day=np.zeros((history_size, 1, 1)),
                                   state=None,
                                   unlocked_non_essential_business_locations=np.zeros((history_size, 1,
                                                                                       num_non_essential_business))
                                   if num_non_essential_business is not None else None,
                                   )

    def update_obs_with_sim_state(self, sim_state: PandemicSimState,
                                  hist_index: int = 0,
                                  business_location_ids: Optional[Sequence[LocationID]] = None) -> None:
        """
        Update the PandemicObservation with the information from PandemicSimState.

        :param sim_state: PandemicSimState instance
        :param hist_index: history time index
        :param business_location_ids: business location ids
        """
        assert hist_index < self.global_infection_summary.shape[0]
        if self.unlocked_non_essential_business_locations is not None and business_location_ids is not None:
            unlocked_non_essential_business_locations = np.asarray([int(not cast(NonEssentialBusinessLocationState,
                                                                                 sim_state.id_to_location_state[
                                                                                     loc_id]).locked)
                                                                    for loc_id in business_location_ids])
            self.unlocked_non_essential_business_locations[hist_index, 0] = unlocked_non_essential_business_locations

        gis = np.asarray([sim_state.global_infection_summary[k] for k in sorted_infection_summary])[None, None, ...]
        gis_a = np.asarray([sim_state.global_infection_summary_alpha[k] for k in sorted_infection_summary])[None, None, ...]
        gis_d = np.asarray([sim_state.global_infection_summary_delta[k] for k in sorted_infection_summary])[None, None, ...]

        self.global_infection_summary[hist_index, 0] = gis / np.sum(gis)
        self.global_infection_summary_alpha[hist_index, 0] = gis_a / np.sum(gis_a)
        self.global_infection_summary_delta[hist_index, 0] = gis_d / np.sum(gis_d)

        gts = np.asarray([sim_state.global_testing_state.summary[k] for k in sorted_infection_summary])[None, None, ...]
        gts_a = np.asarray([sim_state.global_testing_state_alpha.summary[k] for k in sorted_infection_summary])[None, None, ...]
        gts_d = np.asarray([sim_state.global_testing_state_delta.summary[k] for k in sorted_infection_summary])[None, None, ...]

        self.global_testing_summary[hist_index, 0] = gts / np.sum(gts)
        self.global_testing_summary_alpha[hist_index, 0] = gts_a / np.sum(gts_a)
        self.global_testing_summary_delta[hist_index, 0] = gts_d / np.sum(gts_d)

        self.stage[hist_index, 0] = sim_state.regulation_stage

        self.infection_above_threshold[hist_index, 0] = int(sim_state.infection_above_threshold)

        self.time_day[hist_index, 0] = (int(sim_state.sim_time.day) * 24 + int(sim_state.sim_time.hour)) / (365 * 24)

        self.state = sim_state

    @property
    def infection_summary_labels(self) -> Sequence[str]:
        """Return the label for each index in global_infection(or testing)_summary observation entry"""
        return [k.value for k in sorted_infection_summary]
