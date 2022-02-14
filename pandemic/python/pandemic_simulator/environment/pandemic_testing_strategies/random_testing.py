# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import cast

import numpy as np

from ..interfaces import PersonState, InfectionSummary, IndividualInfectionState, PandemicTestResult, PandemicTesting, \
    globals

__all__ = ['RandomPandemicTesting']


class RandomPandemicTesting(PandemicTesting):
    """Implements random pandemic testing based on the specified probabilities."""

    _spontaneous_testing_rate: float
    _symp_testing_rate: float
    _critical_testing_rate: float
    _testing_false_positive_rate: float
    _testing_false_negative_rate: float
    _retest_rate: float
    _numpy_rng: np.random.RandomState

    def __init__(self,
                 spontaneous_testing_rate: float = 1.,
                 symp_testing_rate: float = 1.,
                 critical_testing_rate: float = 1.,
                 testing_false_positive_rate: float = 0.01,
                 testing_false_negative_rate: float = 0.01,
                 retest_rate: float = 0.033):
        """
        :param spontaneous_testing_rate: Testing rate for non symptomatic population.
        :param symp_testing_rate: Testing rate for symptomatic population.
        :param critical_testing_rate: Testing rate for critical population.
        :param testing_false_negative_rate: False negative rate of testing
        :param testing_false_positive_rate: False positive rate of testing
        :param retest_rate: Rate to retest a peron
        """
        self._spontaneous_testing_rate = spontaneous_testing_rate
        self._symp_testing_rate = symp_testing_rate
        self._critical_testing_rate = critical_testing_rate
        self._testing_false_positive_rate = testing_false_positive_rate
        self._testing_false_negative_rate = testing_false_negative_rate
        self._retest_rate = retest_rate
        self._numpy_rng = globals.numpy_rng

    def admit_person(self, person_state: PersonState) -> bool:
        infection_state = cast(IndividualInfectionState, person_state.infection_state)
        infection_state_delta = cast(IndividualInfectionState, person_state.infection_state_delta)
        if infection_state_delta is None:
            infection_state_delta = IndividualInfectionState(summary=InfectionSummary.NONE, spread_probability=0)

        if person_state.test_result == PandemicTestResult.DEAD:
            # A person is not tested if he/she is dead
            return False

        elif infection_state.summary == InfectionSummary.DEAD or infection_state_delta.summary == InfectionSummary.DEAD:
            return True

        rnd = self._numpy_rng.uniform()
        test_person = (
                # if the person is in a hospital, then retest deterministically
                infection_state.is_hospitalized or infection_state_delta.is_hospitalized or 

                # if the person was tested before, then retest based on retest-probability (independent of symptoms)
                (person_state.test_result in {PandemicTestResult.CRITICAL,
                                              PandemicTestResult.POSITIVE} and rnd < self._retest_rate) or

                # if the person shows symptoms, then test based on critical/symptomatic-probability
                (infection_state.shows_symptoms and (
                        (infection_state.summary == InfectionSummary.CRITICAL and rnd < self._critical_testing_rate) or
                        (infection_state.summary != InfectionSummary.CRITICAL and rnd < self._symp_testing_rate))) or

                (infection_state_delta.shows_symptoms and (
                        (infection_state_delta.summary == InfectionSummary.CRITICAL and rnd < self._critical_testing_rate) or
                        (infection_state_delta.summary != InfectionSummary.CRITICAL and rnd < self._symp_testing_rate))) or

                # if the person does not show symptoms, then test based on spontaneous-probability
                (not infection_state.shows_symptoms and rnd < self._spontaneous_testing_rate) or 
                (not infection_state_delta.shows_symptoms and rnd < self._spontaneous_testing_rate)
        )
        return test_person

    def test_person(self, person_state: PersonState) -> PandemicTestResult:
        positive_states = {InfectionSummary.INFECTED, InfectionSummary.CRITICAL}
        infection_state = cast(IndividualInfectionState, person_state.infection_state)
        infection_state_delta = cast(IndividualInfectionState, person_state.infection_state_delta)
        if infection_state_delta is None:
            infection_state_delta = IndividualInfectionState(summary=InfectionSummary.NONE, spread_probability=0)

        ### Get alpha infection test ###
        if infection_state.summary == InfectionSummary.DEAD:
            test_result_alpha = PandemicTestResult.DEAD 

        test_outcome_alpha = infection_state.summary in positive_states
        # account for testing uncertainty
        rnd = self._numpy_rng.uniform()
        if test_outcome_alpha and rnd < self._testing_false_negative_rate:
            test_outcome_alpha = False
        elif not test_outcome_alpha and rnd < self._testing_false_positive_rate:
            test_outcome_alpha = True

        critical_alpha = infection_state.summary == InfectionSummary.CRITICAL
        test_result_alpha = (PandemicTestResult.CRITICAL if test_outcome_alpha and critical_alpha
                       else PandemicTestResult.POSITIVE if test_outcome_alpha else PandemicTestResult.NEGATIVE)

        ### Get delta infection test ###
        if infection_state_delta.summary == InfectionSummary.DEAD:
            test_result_delta = PandemicTestResult.DEAD 

        test_outcome_delta = infection_state_delta.summary in positive_states
        # account for testing uncertainty
        rnd = self._numpy_rng.uniform()
        if test_outcome_delta and rnd < self._testing_false_negative_rate:
            test_outcome_delta = False
        elif not test_outcome_delta and rnd < self._testing_false_positive_rate:
            test_outcome_delta = True

        critical_delta = infection_state_delta.summary == InfectionSummary.CRITICAL
        test_result_delta = (PandemicTestResult.CRITICAL if test_outcome_delta and critical_delta
                       else PandemicTestResult.POSITIVE if test_outcome_delta else PandemicTestResult.NEGATIVE)

        ### Compute overall test result (for summary statistics) ###
        if test_result_alpha == PandemicTestResult.DEAD or test_result_delta == PandemicTestResult.DEAD:
            return PandemicTestResult.DEAD, test_result_alpha, test_result_delta
        if test_result_alpha == PandemicTestResult.CRITICAL or test_result_delta == PandemicTestResult.CRITICAL:
            return PandemicTestResult.CRITICAL, test_result_alpha, test_result_delta
        if test_result_alpha == PandemicTestResult.POSITIVE or test_result_delta == PandemicTestResult.POSITIVE:
            return PandemicTestResult.POSITIVE, test_result_alpha, test_result_delta
        return PandemicTestResult.NEGATIVE, test_result_alpha, test_result_delta
