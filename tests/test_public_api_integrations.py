from qrecon.oracles import (
    FixedPointDeepMLPEqualityLayout,
    FixedPointDeepMLPLayout,
    FixedPointDeepMLPPredicateLayout,
    FixedPointDeepMLPReachabilityCertificate,
    FixedPointDeepMLPResourceBreakdown,
    ReversibleFixedPointDeepMLPEqualityOracle,
    ReversibleFixedPointDeepMLPPredicateOracle,
    ReversibleFixedPointDeepMLPValueOracle,
    compile_structure_preserving_fixed_point_deep_mlp_equality_oracle,
    compile_structure_preserving_fixed_point_deep_mlp_threshold_oracle,
    compile_structure_preserving_fixed_point_deep_mlp_value_oracle,
)
from qrecon.theory import (
    BBHTExistenceDecisionCertificate,
    BBHTExistenceDecisionEvaluation,
    certify_bbht_existence_decision,
    evaluate_bbht_existence_decision,
)


def test_arbitrary_depth_fixed_point_compiler_is_public():
    assert FixedPointDeepMLPLayout is not None
    assert FixedPointDeepMLPPredicateLayout is not None
    assert FixedPointDeepMLPEqualityLayout is not None
    assert FixedPointDeepMLPReachabilityCertificate is not None
    assert FixedPointDeepMLPResourceBreakdown is not None
    assert ReversibleFixedPointDeepMLPValueOracle is not None
    assert ReversibleFixedPointDeepMLPPredicateOracle is not None
    assert ReversibleFixedPointDeepMLPEqualityOracle is not None
    assert callable(compile_structure_preserving_fixed_point_deep_mlp_value_oracle)
    assert callable(compile_structure_preserving_fixed_point_deep_mlp_threshold_oracle)
    assert callable(compile_structure_preserving_fixed_point_deep_mlp_equality_oracle)


def test_zero_solution_bbht_decision_is_public():
    assert BBHTExistenceDecisionCertificate is not None
    assert BBHTExistenceDecisionEvaluation is not None
    assert callable(certify_bbht_existence_decision)
    assert callable(evaluate_bbht_existence_decision)
