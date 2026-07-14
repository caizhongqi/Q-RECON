from qrecon.oracles import (
    ANFOracle,
    ReversibleIntegerAffinePredicateOracle,
    TruthTableOracle,
    audit_anf_oracle,
    audit_structure_preserving_oracle,
    audit_truth_table_oracle,
    build_truth_table_preimage_index,
)


def test_truth_table_artifact_recovers_unique_answer_before_grover():
    oracle = TruthTableOracle.from_function(4, 1, lambda candidate: int(candidate == 11))
    audit = audit_truth_table_oracle(oracle)
    index = build_truth_table_preimage_index(oracle)
    assert index.preimages(1) == (11,)
    assert index.preimages(0) == tuple(candidate for candidate in range(16) if candidate != 11)
    assert audit.reference_evaluations_to_materialize == 16
    assert audit.stores_complete_preimage_information
    assert audit.unique_answer_recoverable_from_artifact
    assert audit.enumerative_setup_exceeds_ideal_grover
    assert oracle.gates[0].required_input == 11


def test_anf_gate_compression_does_not_remove_enumerative_setup():
    truth_table = TruthTableOracle.from_function(5, 1, lambda candidate: int(candidate == 19))
    anf = ANFOracle.from_truth_table(truth_table)
    audit = audit_anf_oracle(anf)
    assert audit.materialized_truth_table_entries == 32
    assert audit.reference_evaluations_to_materialize == 32
    assert audit.unique_answer_recoverable_from_artifact
    assert audit.enumerative_setup_exceeds_ideal_grover
    assert audit.controlled_x_terms == len(anf.gates)


def test_structure_preserving_audit_avoids_truth_table_circularity_only():
    oracle = ReversibleIntegerAffinePredicateOracle(
        (1, -2),
        1,
        input_bits_per_feature=3,
        accumulator_bits=5,
        threshold=0,
        signed_inputs=True,
    )
    audit = audit_structure_preserving_oracle(
        oracle,
        family="integer_affine_threshold",
    )
    assert audit.materialized_truth_table_entries == 0
    assert audit.reference_evaluations_to_materialize == 0
    assert not audit.stores_complete_preimage_information
    assert not audit.unique_answer_recoverable_from_artifact
    assert audit.marked_count is None
    assert "separate reduction" in audit.claim_boundary
    assert audit.toffoli_gates == oracle.resource_estimate(phase_kickback=True).toffoli_gates


def test_preimage_index_validates_output_words():
    oracle = TruthTableOracle.from_function(2, 2, lambda candidate: candidate)
    index = build_truth_table_preimage_index(oracle)
    assert index.preimages(2) == (2,)
    try:
        index.preimages(4)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range output word should be rejected")
