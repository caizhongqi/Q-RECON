from __future__ import annotations

import json

from qrecon.theory.data_loading import (
    explicit_lookup_amortization_report,
    explicit_lookup_description,
    explicit_table_compiler_bit_probe_lower_bound,
    lookup_minterm_resources,
    typical_lookup_circuit_lower_bound,
)


def main() -> None:
    small_table = (0, 1, 3, 7, 15, 31, 63, 127)
    entries = 1024
    word_bits = 256
    payload = {
        "small_exact_minterm_baseline": lookup_minterm_resources(
            small_table, 7
        ).to_dict(),
        "empirical_table_dimensions": explicit_lookup_description(
            entries, word_bits, ancilla_qubits=64
        ).to_dict(),
        "compiler_bit_probe_lower_bound": explicit_table_compiler_bit_probe_lower_bound(
            entries, word_bits
        ),
        "typical_circuit_counting_bound": typical_lookup_circuit_lower_bound(
            entries,
            word_bits,
            ancilla_qubits=64,
            gate_type_count=16,
            max_gate_arity=2,
            exceptional_fraction=0.01,
        ).to_dict(),
        "amortization_example": explicit_lookup_amortization_report(
            entries,
            word_bits,
            classical_setup_cost_per_table_bit=0.0,
            quantum_setup_cost_per_table_bit=1.0,
            classical_variable_cost=1024.0,
            quantum_variable_cost=128.0,
        ).to_dict(),
        "claim_boundary": (
            "An explicit real-data candidate list is not a free coherent oracle. "
            "The setup factor may be set to zero only under an explicit preloaded-QRAM "
            "or externally supplied oracle assumption."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
