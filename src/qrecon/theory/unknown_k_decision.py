from __future__ import annotations

from dataclasses import dataclass

from .unknown_k import (
    BBHTEvaluation,
    BBHTSchedule,
    BBHTUniformCertificate,
    certify_bbht_uniform_success,
    evaluate_bbht_schedule,
)


@dataclass(frozen=True)
class BBHTExistenceDecisionCertificate:
    """One-sided finite certificate for deciding whether any candidate is marked.

    The online rule returns ``present`` only after an exact verification accepts
    a measured candidate; otherwise it returns ``empty`` after the public BBHT
    schedule is exhausted. Therefore the zero-solution case has no false
    positive, while every promised positive marked count inherits the uniform
    BBHT detection guarantee.
    """

    positive_certificate: BBHTUniformCertificate
    zero_case_expected_phase_oracle_calls: float
    zero_case_expected_verification_queries: float

    @property
    def schedule(self) -> BBHTSchedule:
        return self.positive_certificate.schedule

    @property
    def minimum_positive_marked(self) -> int:
        return self.positive_certificate.minimum_marked

    @property
    def certified_positive_detection(self) -> float:
        return self.positive_certificate.certified_minimum_success

    @property
    def false_empty_upper_bound(self) -> float:
        return 1.0 - self.certified_positive_detection

    @property
    def zero_case_success(self) -> float:
        return 1.0

    @property
    def certified_worst_case_decision_success(self) -> float:
        return min(self.zero_case_success, self.certified_positive_detection)

    @property
    def zero_case_expected_total_oracle_calls(self) -> float:
        return (
            self.zero_case_expected_phase_oracle_calls
            + self.zero_case_expected_verification_queries
        )


@dataclass(frozen=True)
class BBHTExistenceDecisionEvaluation:
    certificate: BBHTExistenceDecisionCertificate
    marked: int
    search_evaluation: BBHTEvaluation
    correct_decision_probability: float
    false_empty_probability: float
    false_present_probability: float
    covered_by_certificate: bool

    @property
    def expected_total_oracle_calls(self) -> float:
        return self.search_evaluation.expected_total_oracle_calls


def certify_bbht_existence_decision(
    population: int,
    target_success: float,
    *,
    minimum_positive_marked: int = 1,
    growth_factor: float = 8.0 / 7.0,
    max_rounds: int = 256,
    max_exact_population: int = 4096,
) -> BBHTExistenceDecisionCertificate:
    """Certify a one-sided ``empty`` versus ``present`` decision protocol.

    The promise is ``K = 0`` or ``K >= minimum_positive_marked``. With an exact
    final verifier, ``K = 0`` is always classified correctly because no measured
    candidate can be accepted. For every promised positive ``K``, the returned
    schedule detects and verifies a marked candidate with probability at least
    ``target_success``.
    """

    positive = certify_bbht_uniform_success(
        population,
        target_success,
        minimum_marked=minimum_positive_marked,
        growth_factor=growth_factor,
        max_rounds=max_rounds,
        max_exact_population=max_exact_population,
    )
    zero = evaluate_bbht_schedule(positive.schedule, 0)
    return BBHTExistenceDecisionCertificate(
        positive_certificate=positive,
        zero_case_expected_phase_oracle_calls=zero.expected_phase_oracle_calls,
        zero_case_expected_verification_queries=zero.expected_verification_queries,
    )


def evaluate_bbht_existence_decision(
    certificate: BBHTExistenceDecisionCertificate,
    marked: int,
) -> BBHTExistenceDecisionEvaluation:
    """Evaluate exact finite decision probabilities for a declared marked count."""

    evaluation = evaluate_bbht_schedule(certificate.schedule, marked)
    if marked == 0:
        return BBHTExistenceDecisionEvaluation(
            certificate=certificate,
            marked=0,
            search_evaluation=evaluation,
            correct_decision_probability=1.0,
            false_empty_probability=0.0,
            false_present_probability=0.0,
            covered_by_certificate=True,
        )

    detection = evaluation.achieved_success
    return BBHTExistenceDecisionEvaluation(
        certificate=certificate,
        marked=marked,
        search_evaluation=evaluation,
        correct_decision_probability=detection,
        false_empty_probability=1.0 - detection,
        false_present_probability=0.0,
        covered_by_certificate=marked >= certificate.minimum_positive_marked,
    )
