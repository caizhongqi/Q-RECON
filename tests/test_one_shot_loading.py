import pytest

from qrecon.theory.one_shot_loading import (
    amortized_explicit_table_probe_floor,
    minimum_instances_for_amortized_probe_budget,
    one_shot_explicit_table_boundary,
)


def test_one_shot_explicit_table_has_linear_input_processing_floor():
    report = one_shot_explicit_table_boundary(1024, 256, marked=1)
    assert report.table_description_bits == 1024 * 256
    assert report.exact_quantum_compiler_probe_lower_bound == 1024 * 256
    assert report.classical_full_scan_probe_upper_bound == 1024 * 256
    assert report.ideal_grover_iterations < 1024
    assert report.ideal_grover_success > 0.99
    assert report.rules_out_sublinear_one_shot_input_processing


def test_multiple_marked_items_reduce_queries_but_not_table_input_floor():
    one = one_shot_explicit_table_boundary(256, 8, marked=1)
    many = one_shot_explicit_table_boundary(256, 8, marked=16)
    assert many.ideal_grover_iterations < one.ideal_grover_iterations
    assert many.quantum_one_shot_total_probe_lower_bound == one.quantum_one_shot_total_probe_lower_bound


def test_explicit_table_probe_floor_amortizes_only_with_reuse():
    report = amortized_explicit_table_probe_floor(100, 8, 20)
    assert report.table_description_bits == 800
    assert report.amortized_probe_lower_bound_per_instance == 40.0
    assert minimum_instances_for_amortized_probe_budget(100, 8, 40.0) == 20
    assert minimum_instances_for_amortized_probe_budget(100, 8, 39.9) == 21


def test_invalid_one_shot_boundary_inputs_are_rejected():
    with pytest.raises(ValueError):
        one_shot_explicit_table_boundary(0, 8)
    with pytest.raises(ValueError):
        one_shot_explicit_table_boundary(8, 0)
    with pytest.raises(ValueError):
        one_shot_explicit_table_boundary(8, 8, marked=0)
    with pytest.raises(ValueError):
        amortized_explicit_table_probe_floor(8, 8, 0)
    with pytest.raises(ValueError):
        minimum_instances_for_amortized_probe_budget(8, 8, 0.0)
