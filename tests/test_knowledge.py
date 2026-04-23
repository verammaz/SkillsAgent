"""knowledge plugins and KNOWLEDGE_INJECTION ablation."""

def test_get_knowledge_merges_anomaly_definition():
    from knowledge import get_knowledge

    k = get_knowledge("root_cause_analysis", "Chiller 6", {})
    assert "anomaly_definition" in k
    assert k["anomaly_definition"]["min_context_rows"] == 96
    assert "failure_modes" in k
    assert "sensor_thresholds" in k


def test_knowledge_injection_off_returns_empty(monkeypatch):
    from knowledge import get_knowledge

    monkeypatch.setenv("KNOWLEDGE_INJECTION", "0")
    assert get_knowledge("root_cause_analysis", "Chiller 6", {}) == {}


def test_get_knowledge_empty_skill_name_still_routes_plugins():
    from knowledge import get_knowledge

    # forecasting plugin merge
    k = get_knowledge("forecasting", "Chiller 9", {})
    assert "time_series_metadata" in k
    assert "operating_ranges" in k
