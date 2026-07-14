"""Small executable examples for the Q-RECON theory package."""

from qrecon.theory import (
    AlgorithmCost,
    bayes_reconstruction_success,
    compare_algorithm_costs,
    compare_search_queries,
    minimum_instances_for_quantum_advantage,
)


observation = {
    "candidate-a": "same-gradient",
    "candidate-b": "same-gradient",
    "candidate-c": "different-gradient",
}
prior = {"candidate-a": 0.5, "candidate-b": 0.2, "candidate-c": 0.3}
print("Bayes exact-reconstruction ceiling:", bayes_reconstruction_success(prior, observation))

search = compare_search_queries(population=1024, marked=1, target_success=0.5)
print("Classical queries:", search.classical_queries)
print("Standard-Grover queries:", search.grover_queries)

classical = AlgorithmCost(queries=search.classical_queries or 0, cost_per_query=1.0)
quantum = AlgorithmCost(
    setup_cost=10_000.0,
    fixed_instance_cost=200.0,
    queries=search.grover_queries or 0,
    cost_per_query=20.0,
)
print("One-instance cost:", compare_algorithm_costs(classical, quantum))
print("Break-even instances:", minimum_instances_for_quantum_advantage(classical, quantum))
