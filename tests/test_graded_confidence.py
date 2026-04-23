"""Graded `score_diagnosis_confidence` produces a continuous distribution.

Paper's θ sweep (Table 5, condition E) expects a curve, not a step, so the
scoring function must combine match quality, severity/density, coverage, TSAD
corroboration, and WO history — not just a 6-bucket lookup.
"""

import pytest


def _hi_fmsr_meta():
    return {"failure": "compressor_overheating", "matched_via": "fmsr"}


def _unknown_meta():
    return {"failure": "unknown_failure", "matched_via": "unknown"}


def test_back_compat_two_arg_call_still_works():
    from tools import score_diagnosis_confidence

    # legacy two-arg call must not raise
    c = score_diagnosis_confidence(
        {"severity": "high", "anomaly_details": ["s1 above max"]},
        _hi_fmsr_meta(),
    )
    assert 0.05 <= c <= 0.98


def test_unknown_failure_gets_very_low_confidence():
    from tools import score_diagnosis_confidence

    c = score_diagnosis_confidence(
        {"severity": "none", "anomaly_details": []},
        _unknown_meta(),
    )
    assert c <= 0.4, f"unknown/none should be low but got {c}"


def test_severity_monotone_at_fixed_coverage():
    """high > medium > low > none when other signals held constant."""
    from tools import score_diagnosis_confidence

    def _score(sev, n_details):
        return score_diagnosis_confidence(
            {"severity": sev, "anomaly_details": [f"s{i} above max" for i in range(n_details)]},
            _hi_fmsr_meta(),
        )

    c_high = _score("high", 3)
    c_med = _score("medium", 3)
    c_low = _score("low", 3)
    c_none = _score("none", 0)
    assert c_high > c_med > c_low > c_none


def test_density_raises_confidence():
    """More anomaly_details → higher confidence (log-scaled, saturating)."""
    from tools import score_diagnosis_confidence

    def _score(n):
        return score_diagnosis_confidence(
            {"severity": "medium", "anomaly_details": [f"s{i} above max" for i in range(n)]},
            _hi_fmsr_meta(),
        )

    assert _score(10) > _score(1) > _score(0)


def test_coverage_peaks_at_about_30_percent():
    """With 10 sensors, 3 affected > 8 affected (systemic) and > 0 affected."""
    from tools import score_diagnosis_confidence

    readings = {f"sensor_{i}": [0.0] * 100 for i in range(10)}
    meta = _hi_fmsr_meta()

    def _conf(affected):
        details = [f"sensor_{i} above max" for i in range(affected)]
        return score_diagnosis_confidence(
            {"severity": "high", "anomaly_details": details},
            meta,
            sensor_data={"readings": readings},
        )

    c_30pct = _conf(3)
    c_sparse = _conf(0)
    c_systemic = _conf(9)
    assert c_30pct > c_sparse
    assert c_30pct > c_systemic


def test_tsad_corroboration_raises_confidence():
    """Deep TSFM's integrated-TSAD records should boost the post-deep score."""
    from tools import score_diagnosis_confidence

    base = {"severity": "medium", "anomaly_details": ["s1 above max"]}
    with_tsad = {**base, "tsfm_integrated_tsad_records": 18}
    c_base = score_diagnosis_confidence(base, _hi_fmsr_meta())
    c_tsad = score_diagnosis_confidence(with_tsad, _hi_fmsr_meta())
    assert c_tsad > c_base


def test_wo_history_match_adds_confidence():
    from tools import score_diagnosis_confidence

    anomaly = {"severity": "medium", "anomaly_details": ["s1 above max"]}
    meta = {"failure": "compressor_overheating", "matched_via": "fmsr"}
    c_no_wo = score_diagnosis_confidence(anomaly, meta)
    c_wo = score_diagnosis_confidence(
        anomaly,
        meta,
        wo_history=[{"failure_code": "compressor overheating", "date": "2024-03-01"}],
    )
    assert c_wo > c_no_wo


def test_graded_distribution_spans_multiple_buckets():
    """Across realistic (severity, via, density) combos, we get >4 distinct values
    in [0.05, 0.98] — the old 6-bucket lookup would have collapsed to at most 6."""
    from tools import score_diagnosis_confidence

    combos = [
        ({"severity": "none", "anomaly_details": []}, _unknown_meta()),
        ({"severity": "low", "anomaly_details": ["s1"]}, {"failure": "fm", "matched_via": "knowledge"}),
        ({"severity": "medium", "anomaly_details": ["s1", "s2"]}, {"failure": "fm", "matched_via": "fmsr"}),
        ({"severity": "high", "anomaly_details": ["s1", "s2", "s3"]}, {"failure": "fm", "matched_via": "fmsr"}),
        ({"severity": "high", "anomaly_details": ["s1"] * 8, "tsfm_integrated_tsad_records": 15},
         {"failure": "fm", "matched_via": "fmsr"}),
        ({"severity": "medium", "anomaly_details": ["s1", "s2"]},
         {"failure": "fm", "matched_via": "fmsr"}),
    ]
    scores = {round(score_diagnosis_confidence(a, m), 3) for a, m in combos}
    assert len(scores) >= 4, f"expected ≥4 distinct scores, got {sorted(scores)}"
    assert all(0.05 <= s <= 0.98 for s in scores)


def test_task_specificity_lifts_matching_prompt():
    """A prompt that names the diagnosed subsystem scores higher than a vague one,
    all other signals held fixed."""
    from tools import score_diagnosis_confidence

    anomaly = {"severity": "medium", "anomaly_details": ["s1 above max"]}
    meta = {"failure": "vibration_fault", "matched_via": "fmsr"}
    c_vague = score_diagnosis_confidence(anomaly, meta, task="Something is off.")
    c_specific = score_diagnosis_confidence(
        anomaly, meta, task="Chiller 9 vibration has been rising."
    )
    assert c_specific > c_vague


def test_task_specificity_without_task_is_neutral_not_zero():
    """Missing task argument must not zero out the signal (back-compat)."""
    from tools import score_diagnosis_confidence

    anomaly = {"severity": "medium", "anomaly_details": ["s1 above max"]}
    meta = {"failure": "vibration_fault", "matched_via": "fmsr"}
    c_no_task = score_diagnosis_confidence(anomaly, meta)
    c_vague = score_diagnosis_confidence(anomaly, meta, task="hi")
    # Both are in the low/neutral band (0.3) — differ by <0.02 from reweighting.
    assert abs(c_no_task - c_vague) < 0.05


def test_task_specificity_distinct_scores_across_task_bank():
    """Running the 12-task bank through the scorer produces ≥3 distinct pre-deep
    confidences, so the θ sweep has a real curve instead of a step."""
    from scenario_loader import BUILTIN_TASK_BANK
    from tools import score_diagnosis_confidence

    anomaly = {"severity": "medium", "anomaly_details": ["s1 above max", "s2 low"]}
    meta = {"failure": "vibration_fault", "matched_via": "fmsr"}

    scores = {
        round(score_diagnosis_confidence(anomaly, meta, task=task), 3)
        for _, task, _ in BUILTIN_TASK_BANK
    }
    assert len(scores) >= 3, f"expected ≥3 distinct task-driven scores, got {sorted(scores)}"


def test_sweep_boundary_is_graded_not_step():
    """θ ∈ {0.5, 0.6, 0.7, 0.8} should give different invocation decisions for
    confidences drawn from the graded scorer (not all identical)."""
    from tools import score_diagnosis_confidence
    from confidence_evaluator import should_invoke_deep_tsfm

    combos = [
        ({"severity": "none", "anomaly_details": []}, _unknown_meta()),
        ({"severity": "medium", "anomaly_details": ["s1"]},
         {"failure": "fm", "matched_via": "knowledge"}),
        ({"severity": "high", "anomaly_details": ["s1", "s2"]},
         {"failure": "fm", "matched_via": "fmsr"}),
        ({"severity": "high", "anomaly_details": ["s1"] * 8, "tsfm_integrated_tsad_records": 15},
         {"failure": "fm", "matched_via": "fmsr"}),
    ]
    scores = [score_diagnosis_confidence(a, m) for a, m in combos]

    decisions_by_theta = {
        t: [should_invoke_deep_tsfm(s, theta=t, conditional_enabled=True) for s in scores]
        for t in (0.5, 0.6, 0.7, 0.8)
    }
    # Not all thetas produce identical decision vectors → curve, not step
    distinct = {tuple(v) for v in decisions_by_theta.values()}
    assert len(distinct) >= 2, (
        f"expected varied invocation across θ ∈ 0.5..0.8 but got {decisions_by_theta}"
    )
