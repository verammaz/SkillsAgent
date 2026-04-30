"""Confidence-Evaluator gate (proposal §5).

Decides whether to invoke deep statistical TSFM after FMSR's first diagnosis.
The numerical confidence itself comes from ``tools.score_diagnosis_confidence``.
"""

from __future__ import annotations

import os


def theta_from_env() -> float:
    """Deep-TSFM threshold θ. Default 0.85; ``eval_runner`` sweeps 0.5–0.95."""
    return float(os.getenv("RCA_CONFIDENCE_THETA", "0.85"))


def conditional_deep_tsfm_enabled() -> bool:
    return os.getenv("ENABLE_CONDITIONAL_DEEP_TSFM", "1").lower() in ("1", "true", "yes")


def always_deep_tsfm_from_env() -> bool:
    """If True, RCA runs ``deep_tsfm_refine_anomalies`` whenever deep TSFM is enabled.

    Skips the θ gate (``RCA_CONFIDENCE_THETA``). Used for ablation condition F
    (skills + knowledge, always deep) vs condition E (θ-gated).
    """
    return os.getenv("RCA_ALWAYS_DEEP_TSFM", "0").lower() in ("1", "true", "yes")


def should_invoke_deep_tsfm(
    diagnosis_confidence: float,
    *,
    theta: float | None = None,
    conditional_enabled: bool | None = None,
) -> bool:
    """Return True iff deep TSFM should run (proposal Alg. 2).

    ``confidence >= theta`` → skip expensive TSFM.
    ``confidence < theta``  → run ``deep_tsfm_refine_anomalies`` and re-map.
    """
    if conditional_enabled is None:
        conditional_enabled = conditional_deep_tsfm_enabled()
    if not conditional_enabled:
        return False
    if always_deep_tsfm_from_env():
        return True
    t = theta if theta is not None else theta_from_env()
    return diagnosis_confidence < t
