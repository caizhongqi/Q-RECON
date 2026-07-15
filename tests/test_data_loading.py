import pytest

from qrecon.theory.data_loading import (
    explicit_lookup_amortization_report,
    explicit_lookup_description,
    explicit_table_compiler_bit_probe_lower_bound,
    lookup_minterm_resources,
    typical_lookup_circuit_lower_bound,
)


def test_explicit_description_and_compiler_probe_lower_bound():
    description = explicit_lookup_description(5, 7, ancilla_qubits=3)
    assert description.index_bits == 3
    assert description.table_description_bits == 35
    assert description.total_circuit_qubits == 13
    assert explicit_table_compiler_bit_probe_lower_bound(5, 7) == 35


def test_naive_minterm_lookup_resources_are_auditable():
    report = lookup_minterm_resources([0, 1, 2, 3], 2)
    assert report.index_bits == 2
    assert report.table_description_bits == 8
    assert report.nonzero_entries == 3
    assert report.output_one_bits == 4
    assert report.x_gates == 4
    assert report.toffoli_gates == 4
    assert report.cnot_gates == 0
    assert report.logical_qubits == 4


def test_typical_counting_bound_grows_with_table_size():
    small = typical_lookup_circuit_lower_bound(16, 4, exceptional_fraction=0.01)
    large = typical_lookup_circuit_lower_bound(256, 8, exceptional_fraction=0.01)
    assert small.minimum_gate_count > 0
    assert large.minimum_gate_count > small.minimum_gate_count
    assert small.short_circuit_fraction_upper_bound <= 0.01
    assert large.short_circuit_fraction_upper_bound <= 0.01


def test_more_workspace_weakens_the_counting_lower_bound():
    compact = typical_lookup_circuit_lower_bound(128, 8, ancilla_qubits=0)
    roomy = typical_lookup_circuit_lower_bound(128, 8, ancilla_qubits=128)
    assert roomy.minimum_gate_count <= compact.minimum_gate_count


def test_explicit_table_setup_requires_amortization():
    report = explicit_lookup_amortization_report(
        100,
        8,
        quantum_setup_cost_per_table_bit=1.0,
        classical_variable_cost=100.0,
        quantum_variable_cost=20.0,
    )
    assert report.table_description_bits == 800
    assert report.quantum_setup_cost == 800.0
    assert report.minimum_instances_for_advantage == 11


def test_invalid_lookup_inputs_are_rejected():
    with pytest.raises(ValueError):
        explicit_lookup_description(0, 8)
    with pytest.raises(ValueError):
        typical_lookup_circuit_lower_bound(4, 4, exceptional_fraction=1.0)
    with pytest.raises(ValueError):
        lookup_minterm_resources([4], 2)
