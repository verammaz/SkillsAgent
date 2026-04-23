"""skills.root_cause_analysis: conditional deep TSFM + knowledge wiring."""

from unittest.mock import patch


def test_rca_invokes_deep_when_confidence_low(monkeypatch):
    monkeypatch.setenv("ENABLE_CONDITIONAL_DEEP_TSFM", "1")
    monkeypatch.setenv("RCA_CONFIDENCE_THETA", "0.8")

    def fake_deep(aid, sd, anom, anomaly_definition=None):
        return {
            **anom,
            "anomaly_details": (anom.get("anomaly_details") or []) + ["deep_stub"],
            "anomalies_detected": True,
            "severity": "medium",
            "deep_tsfm_refined": True,
        }

    with patch("skills.score_diagnosis_confidence", side_effect=[0.25, 0.9]):
        with patch("skills.deep_tsfm_refine_anomalies", side_effect=fake_deep):
            with patch("skills.map_failure_with_meta") as mf:
                mf.side_effect = [
                    {"failure": "x", "matched_via": "unknown"},
                    {"failure": "y", "matched_via": "fmsr"},
                ]
                from skills import root_cause_analysis

                ctx = {}
                out = root_cause_analysis("Chiller 6", ctx, "fault task")
                assert out["output"]["deep_tsfm_invoked"] is True
                assert "deep_stub" in out["output"]["anomaly_analysis"]["anomaly_details"]
                assert mf.call_count == 2


def test_rca_skips_deep_when_confidence_high(monkeypatch):
    monkeypatch.setenv("ENABLE_CONDITIONAL_DEEP_TSFM", "1")
    monkeypatch.setenv("RCA_CONFIDENCE_THETA", "0.8")

    with patch("skills.deep_tsfm_refine_anomalies") as deep:
        with patch("skills.map_failure_with_meta", return_value={"failure": "f", "matched_via": "fmsr"}):
            with patch("skills.score_diagnosis_confidence", return_value=0.95):
                from skills import root_cause_analysis

                ctx = {}
                out = root_cause_analysis("Chiller 6", ctx, "fault task")
                assert out["output"]["deep_tsfm_invoked"] is False
                deep.assert_not_called()


def test_deep_tsfm_invokes_when_medium_severity_and_theta_default(monkeypatch):
    """High-severity mock runs skip deep; medium (0.84) is below default θ=0.85."""
    monkeypatch.setenv("ENABLE_CONDITIONAL_DEEP_TSFM", "1")
    monkeypatch.delenv("RCA_CONFIDENCE_THETA", raising=False)
    from confidence_evaluator import theta_from_env

    assert theta_from_env() == 0.85
    with patch("skills.deep_tsfm_refine_anomalies") as deep:
        with patch("skills.map_failure_with_meta", return_value={"failure": "f", "matched_via": "fmsr"}):
            with patch("skills.score_diagnosis_confidence", return_value=0.84):
                from skills import root_cause_analysis

                ctx = {}
                out = root_cause_analysis("Chiller 6", ctx, "fault task")
                assert out["output"]["deep_tsfm_invoked"] is True
                deep.assert_called_once()


def test_forecasting_fetches_sensor_data_when_missing():
    from skills import forecasting

    ctx = {}
    out = forecasting("Chiller 6", ctx, "forecast task")
    assert "sensor_data" in ctx
    assert "forecast" in out["output"]
    assert out["output"]["forecast"].get("source") == "mock"
