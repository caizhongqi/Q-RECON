import numpy as np
import pytest

from qrecon.benchmarks.tensor_candidates import audit_quantized_candidate_tensor


def test_quantization_reports_new_collisions_and_bayes_ceiling():
    values = np.array([[0.1], [0.2], [1.0]], dtype=np.float32)
    report = audit_quantized_candidate_tensor(values, 3, 0)
    assert report.codes.reshape(-1).tolist() == [0, 0, 1]
    assert report.original_unique_candidate_count == 3
    assert report.quantized_unique_candidate_count == 2
    assert report.quantization_induced_collision_count == 1
    assert report.exact_index_bayes_success_before_quantization_uniform == 1.0
    assert report.exact_index_bayes_success_after_quantization_uniform == pytest.approx(2 / 3)
    assert report.loading.exact_index_bayes_success_uniform == pytest.approx(2 / 3)


def test_half_ties_round_away_from_zero():
    report = audit_quantized_candidate_tensor(
        np.array([[0.5], [-0.5]], dtype=np.float64), 3, 0
    )
    assert report.codes.reshape(-1).tolist() == [1, -1]
    assert report.maximum_absolute_error == pytest.approx(0.5)


def test_saturation_is_explicit_and_raise_never_wraps():
    values = np.array([[3.0], [-3.0]])
    with pytest.raises(OverflowError):
        audit_quantized_candidate_tensor(values, 2, 0, overflow="raise")
    saturated = audit_quantized_candidate_tensor(values, 2, 0, overflow="saturate")
    assert saturated.codes.reshape(-1).tolist() == [1, -2]
    assert saturated.saturation_count == 2


def test_source_hash_is_deterministic_and_dtype_sensitive():
    left = audit_quantized_candidate_tensor(np.array([[0.0, 1.0]], dtype=np.float32), 4, 1)
    same = audit_quantized_candidate_tensor(np.array([[0.0, 1.0]], dtype=np.float32), 4, 1)
    wider = audit_quantized_candidate_tensor(np.array([[0.0, 1.0]], dtype=np.float64), 4, 1)
    assert left.source_sha256 == same.source_sha256
    assert left.source_sha256 != wider.source_sha256


def test_nonfinite_and_invalid_width_inputs_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        audit_quantized_candidate_tensor(np.array([[np.nan]]), 4, 1)
    with pytest.raises(ValueError, match="62"):
        audit_quantized_candidate_tensor(np.array([[0.0]]), 63, 1)
