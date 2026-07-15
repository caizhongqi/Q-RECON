from .analytic_linear import infer_class_label_from_last_bias, invert_first_linear_gradient
from .gradient_inversion import (
    AttackResult,
    GradientInversionAttack,
    gradient_matching_loss,
    leak_gradients,
)
from .head_representation import (
    HeadRepresentationAttackResult,
    HeadRepresentationInversionAttack,
    LinearHeadLeakageReport,
    capture_final_linear_input,
    find_last_biased_linear,
    recover_single_effective_head_input,
)
from .time_series_regularization import (
    linear_trend_penalty,
    periodicity_penalty,
    resolution_consistency_penalty,
)

__all__ = [
    "AttackResult",
    "GradientInversionAttack",
    "HeadRepresentationAttackResult",
    "HeadRepresentationInversionAttack",
    "LinearHeadLeakageReport",
    "capture_final_linear_input",
    "find_last_biased_linear",
    "gradient_matching_loss",
    "leak_gradients",
    "recover_single_effective_head_input",
    "invert_first_linear_gradient",
    "infer_class_label_from_last_bias",
    "linear_trend_penalty",
    "periodicity_penalty",
    "resolution_consistency_penalty",
]
