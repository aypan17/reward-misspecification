# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Tuple, cast

from .contact_tracer import ContactTracer
from .ids import PersonID, LocationID
from .infection_model import IndividualInfectionState, Risk, InfectionSummary
from .pandemic_testing_result import PandemicTestResult
from .pandemic_types import NoOP
from .regulation import PandemicRegulation
from .sim_time import SimTime

__all__ = ['Person', 'PersonState', 'get_infection_summary']


@dataclass
class PersonState:
    """State of the person."""
    current_location: LocationID
    risk: Risk
    infection_state: Optional[IndividualInfectionState] = None
    infection_spread_multiplier: float = 1.

    infection_state_delta: Optional[IndividualInfectionState] = None
    infection_spread_multiplier_delta: float = 1.5

    quarantine: bool = field(init=False, default=False)
    quarantine_if_contact_positive: bool = field(init=False, default=False)
    quarantine_if_household_quarantined: bool = field(init=False, default=False)
    sick_at_home: bool = field(init=False, default=False)
    avoid_gathering_size: int = field(init=False, default=-1)

    test_result: PandemicTestResult = field(init=False, default=PandemicTestResult.UNTESTED)
    test_result_alpha: PandemicTestResult = field(init=False, default=PandemicTestResult.UNTESTED)
    test_result_delta: PandemicTestResult = field(init=False, default=PandemicTestResult.UNTESTED)

    avoid_location_types: List[type] = field(default_factory=list, init=False)
    not_infection_probability: float = field(default=1., init=False)
    not_infection_probability_delta: float = field(default=1., init=False)
    not_infection_probability_history: List[Tuple[LocationID, float]] = field(default_factory=list, init=False)
    not_infection_probability_delta_history: List[Tuple[LocationID, float]] = field(default_factory=list, init=False)


def get_infection_summary(person_state: PersonState) -> InfectionSummary:
    if person_state.infection_state is not None or person_state.infection_state_delta is not None:
        state = cast(IndividualInfectionState, person_state.infection_state)
        state_delta = cast(IndividualInfectionState, person_state.infection_state_delta)
        if state_delta is None:
            return state.summary
        if state.summary == InfectionSummary.NONE and state_delta.summary == InfectionSummary.NONE:
            return InfectionSummary.NONE
        elif state.summary == InfectionSummary.DEAD or state_delta.summary == InfectionSummary.DEAD:
            return InfectionSummary.DEAD
        elif state.summary == InfectionSummary.CRITICAL or state_delta.summary == InfectionSummary.CRITICAL:
            return InfectionSummary.CRITICAL
        elif state.summary == InfectionSummary.INFECTED or state_delta.summary == InfectionSummary.INFECTED:
            return InfectionSummary.INFECTED 
        else:
            return InfectionSummary.RECOVERED
    return None


class Person(ABC):
    """Class that implements a sim person automaton with a pre-defined policy."""

    @abstractmethod
    def step(self, sim_time: SimTime, contact_tracer: Optional[ContactTracer] = None) -> Optional[NoOP]:
        """
        Method that steps through the person's policy. The step can return a
        NoOp to indicate no operation was carried out.

        :param sim_time: Current simulation time.
        :param contact_tracer: Traces of previous contacts of the person.
        :return: Return NoOp if no operation was carried out otherwise None.
        """
        pass

    @abstractmethod
    def receive_regulation(self, regulation: PandemicRegulation) -> None:
        """
        Receive a regulation that can potentially update the person's policy.

        :param regulation: a PandemicRegulation instance
        """
        pass

    @abstractmethod
    def enter_location(self, location_id: LocationID) -> bool:
        """
        Enter a location.

        :param location_id: LocationID instance
        :return: True if successful else False
        """
        pass

    @property
    @abstractmethod
    def id(self) -> PersonID:
        """
        Method that returns the id of the person.

        :return: ID of the person.
        """
        pass

    @property
    @abstractmethod
    def home(self) -> LocationID:
        """
        Property that returns the person's home location id.

        :return: ID of the home location
        """
        pass

    @property
    @abstractmethod
    def assigned_locations(self) -> Sequence[LocationID]:
        """
        Property that returns a sequence of location ids that the person is assigned to.

        :return: A collection of LocationIDs
        """

    @property
    @abstractmethod
    def state(self) -> PersonState:
        """
        Property that returns the current state of the person.

        :return: Current state of the person.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset person to its initial state."""
