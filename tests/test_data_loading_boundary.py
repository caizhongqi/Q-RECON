import pytest

from qrecon.theory.data_loading_boundary import (
    PositiveIntegerWorkloadRegion,
    certified_explicit_table_no_advantage_region,
    certify_explicit_table_no_advantage,
)


def test_explicit_table_compilation_can_certify_one_shot_no_advantage():
    report = certify_explicit_table_no_advantage(
        1024,
        32,
        classical_variable_upper_bound=20_000,
    )
    assert report.table_description_bits == 32_768
    assert report.quantum_setup_lower_bound == 32_768
    assert report.quantum_total_lower_bound == 32_768
    assert report.classical_total_upper_bound == 20_000
    assert report.certified_no_advantage
    assert report.no_advantage_margin == 12_768

    unresolved = certify_explicit_table_no_advantage(
        1024,
        32,
        classical_variable_upper_bound=40_000,
    )
    assert not unresolved.certified_no_advantage
    assert unresolved.no_advantage_margin < 0


def test_no_advantage_region_shows_setup_amortization_threshold():
    region = certified_explicit_table_no_advantage_region(
        100,
        1,
        quantum_variable_lower_bound=1,
        classical_variable_upper_bound=10,
    )
    assert region == PositiveIntegerWorkloadRegion(1, 11)
    assert region.contains(1)
    assert region.contains(11)
    assert not region.contains(12)


def test_no_advantage_region_can_be_unbounded_or_start_late():
    all_workloads = certified_explicit_table_no_advantage_region(
        16,
        8,
        quantum_variable_lower_bound=10,
        classical_variable_upper_bound=5,
    )
    assert all_workloads == PositiveIntegerWorkloadRegion(1, None)
    assert all_workloads.contains(1_000_000)

    late = certified_explicit_table_no_advantage_region(
        1,
        1,
        quantum_setup_extra_lower_bound=0,
        classical_setup_upper_bound=101,
        quantum_variable_lower_bound=11,
        classical_variable_upper_bound=1,
    )
    assert late == PositiveIntegerWorkloadRegion(10, None)
    assert not late.contains(9)
    assert late.contains(10)


def test_no_advantage_region_can_be_empty():
    assert (
        certified_explicit_table_no_advantage_region(
            1,
            1,
            classical_setup_upper_bound=10,
            quantum_variable_lower_bound=0,
            classical_variable_upper_bound=1,
        )
        is None
    )


def test_explicit_table_boundary_validates_costs_and_workload():
    with pytest.raises(ValueError):
        certify_explicit_table_no_advantage(
            2,
            2,
            instances=0,
            classical_variable_upper_bound=1,
        )
    with pytest.raises(ValueError):
        certify_explicit_table_no_advantage(
            2,
            2,
            quantum_variable_lower_bound=-1,
            classical_variable_upper_bound=1,
        )
    with pytest.raises(ValueError):
        certified_explicit_table_no_advantage_region(
            2,
            2,
            classical_variable_upper_bound=float("inf"),
        )
