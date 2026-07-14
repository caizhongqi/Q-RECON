"""Executable information and query-complexity bounds for Q-RECON."""

from .bounds import (
    all_pairs_epsilon_private_uniform_bound,
    bayes_reconstruction_success,
    binary_helstrom_success,
    channel_bayes_reconstruction_success,
    conditional_min_entropy_bits,
    observation_fibres,
    postprocess_channel,
    uniform_fibre_success,
)
from .costs import (
    AlgorithmCost,
    CostComparison,
    compare_algorithm_costs,
    maximum_quantum_query_cost_for_advantage,
    minimum_instances_for_quantum_advantage,
    oracle_error_success_lower_bound,
)
from .search import (
    SearchComparison,
    classical_queries_for_success,
    classical_success_without_replacement,
    compare_search_queries,
    expected_classical_queries,
    grover_queries_for_success,
    grover_success,
    optimal_standard_grover_iterations,
)

__all__ = [
    "AlgorithmCost",
    "CostComparison",
    "SearchComparison",
    "all_pairs_epsilon_private_uniform_bound",
    "bayes_reconstruction_success",
    "binary_helstrom_success",
    "channel_bayes_reconstruction_success",
    "classical_queries_for_success",
    "classical_success_without_replacement",
    "compare_algorithm_costs",
    "compare_search_queries",
    "conditional_min_entropy_bits",
    "expected_classical_queries",
    "grover_queries_for_success",
    "grover_success",
    "maximum_quantum_query_cost_for_advantage",
    "minimum_instances_for_quantum_advantage",
    "observation_fibres",
    "optimal_standard_grover_iterations",
    "oracle_error_success_lower_bound",
    "postprocess_channel",
    "uniform_fibre_success",
]
