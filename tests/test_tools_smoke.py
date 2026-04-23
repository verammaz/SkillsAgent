"""tools.py: forecast, deep_tsfm, detect_anomaly with mocks (no AssetOpsBench)."""

def test_forecast_sensor_mock_without_sensor_data():
    from tools import forecast_sensor

    r = forecast_sensor("Chiller 6", "flow_rate_GPM", horizon_days=5)
    assert r["source"] == "mock"
    assert len(r["forecasted"]) == 5


def test_forecast_sensor_respects_horizon_cap(monkeypatch):
    from tools import forecast_sensor

    monkeypatch.setenv("TSFM_MAX_FORECAST_STEPS", "10")
    readings = {"x": [float(i) for i in range(120)]}
    r = forecast_sensor(
        "Chiller 6",
        "x",
        horizon_days=100,
        sensor_data={"readings": readings},
    )
    assert r["source"] == "mock"
    assert len(r["forecasted"]) == 10


def test_deep_tsfm_refine_accepts_anomaly_definition():
    from tools import deep_tsfm_refine_anomalies, detect_anomaly

    readings = {"flow_rate_GPM": [float(i) for i in range(120)]}
    sd = {"readings": readings}
    an = detect_anomaly(sd)
    out = deep_tsfm_refine_anomalies(
        "Chiller 6",
        sd,
        an,
        anomaly_definition={"min_context_rows": 96},
    )
    assert out.get("deep_tsfm_refined") is True


def test_detect_anomaly_uses_profiles():
    from tools import detect_anomaly

    data = {
        "readings": {
            "vibration_mm_s": [1.0, 1.0, 10.0],
        }
    }
    r = detect_anomaly(data)
    assert r["anomalies_detected"] is True
