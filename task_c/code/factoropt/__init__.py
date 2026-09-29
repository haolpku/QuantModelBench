"""Task C: executable LLM-generated multi-factor model evaluation.

The package intentionally keeps model execution and portfolio accounting out
of the LLM adapter.  A candidate source is validated, executed locally, and
only then passed to the deterministic JKP-compatible backtester.
"""

from .task_c import (
    CODE_RESPONSE_SCHEMA_VERSION,
    PAYLOAD_SCHEMA_VERSION,
    TRACKS,
    LEGACY_TRACKS,
    build_code_generation_payload,
    build_rolling_windows,
    code_response_hash,
    materialize_split,
    stable_hash,
    validate_code_response,
    validate_public_payload,
    validation_feedback,
    validation_metrics,
)
from .semantic import (
    ALL_TRACKS,
    CANONICAL_TRACKS,
    SEMANTIC_CARD_SCHEMA_VERSION,
    normalize_track,
    semantic_card_payload,
    semantic_cards_payload,
    validate_semantic_cards,
)
from .task_c_code_runner import (
    CodeExecutionResult,
    check_code_policy,
    run_candidate,
)
from .task_c_backtest import (
    BacktestDataError,
    DELISTING_RETURN_POLICY,
    RETURN_POLICY,
    backtest_predictions,
    backtest_weights,
    drift_weights,
    form_jkp_weights,
    load_daily_returns,
    load_stock_panel,
    performance_metrics,
    settle_daily_returns,
)
from .task_c_iterative import (
    RevisionConfig,
    ValidationBacktest,
    evaluate_candidate_on_validation,
    iterative_revision_loop,
    rolling_predict,
)
from .task_c_data import assert_point_in_time, mask_identifiers, point_in_time_audit

__all__ = [
    "CODE_RESPONSE_SCHEMA_VERSION",
    "PAYLOAD_SCHEMA_VERSION",
    "TRACKS",
    "LEGACY_TRACKS",
    "ALL_TRACKS",
    "CANONICAL_TRACKS",
    "SEMANTIC_CARD_SCHEMA_VERSION",
    "normalize_track",
    "semantic_card_payload",
    "semantic_cards_payload",
    "validate_semantic_cards",
    "build_code_generation_payload",
    "build_rolling_windows",
    "code_response_hash",
    "materialize_split",
    "stable_hash",
    "validate_code_response",
    "validate_public_payload",
    "validation_feedback",
    "validation_metrics",
    "CodeExecutionResult",
    "check_code_policy",
    "run_candidate",
    "BacktestDataError",
    "DELISTING_RETURN_POLICY",
    "RETURN_POLICY",
    "backtest_predictions",
    "backtest_weights",
    "drift_weights",
    "form_jkp_weights",
    "load_daily_returns",
    "load_stock_panel",
    "performance_metrics",
    "settle_daily_returns",
    "RevisionConfig",
    "ValidationBacktest",
    "evaluate_candidate_on_validation",
    "iterative_revision_loop",
    "rolling_predict",
    "assert_point_in_time",
    "mask_identifiers",
    "point_in_time_audit",
]
