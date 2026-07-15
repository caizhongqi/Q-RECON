import numpy as np
import pytest

from qrecon.benchmarks.candidate_loading import empirical_candidate_loading_report


def test_empirical_report_hashes_and_audits_duplicate_fibres():
    candidates = np.array(
        [
            [[-1, 0], [1, 2]],
            [[-1, 0], [1, 2]],
            [[2, 1], [0, -1]],
        ],
        dtype=np.int64,
    )
    first = empirical_candidate_loading_report(candidates, 3)
    second = empirical_candidate_loading_report(candidates.copy(), 3)
    assert first.source_sha256 == second.source_sha256
    assert first.candidate_count == 3
    assert first.unique_candidate_count == 2
    assert first.duplicate_candidate_count == 1
    assert first.candidate_shape == (2, 2)
    assert first.word_bits == 12
    assert first.exact_index_bayes_success_uniform == pytest.approx(2 / 3)
    assert first.explicit_lookup.table_description_bits == 36
    assert first.deduplicated_lookup.table_description_bits == 24
    assert first.minterm_resources is not None


def test_hash_changes_with_order_or_content():
    candidates = np.array([[0, 1], [2, 3]], dtype=np.int64)
    original = empirical_candidate_loading_report(candidates, 3)
    reordered = empirical_candidate_loading_report(candidates[::-1], 3)
    changed = empirical_candidate_loading_report(np.array([[0, 1], [2, 2]]), 3)
    assert original.source_sha256 != reordered.source_sha256
    assert original.source_sha256 != changed.source_sha256


def test_large_table_skips_literal_minterm_backend_but_keeps_bounds():
    candidates = np.arange(64, dtype=np.int64).reshape(8, 8)
    report = empirical_candidate_loading_report(
        candidates,
        6,
        signed=False,
        max_minterm_table_bits=32,
    )
    assert report.minterm_resources is None
    assert report.minterm_skipped_reason is not None
    assert report.compiler_bit_probe_lower_bound == 8 * 8 * 6
    assert report.typical_circuit_lower_bound.minimum_gate_count > 0


def test_invalid_candidate_codes_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        empirical_candidate_loading_report(np.array([1, 2, 3]), 3)
    with pytest.raises(ValueError, match="declared format"):
        empirical_candidate_loading_report(np.array([[4]]), 3, signed=True)
    with pytest.raises(ValueError, match="integers"):
        empirical_candidate_loading_report(np.array([[0.5, 1.0]]), 3)
