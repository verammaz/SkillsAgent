"""eval_runner ablation entrypoints (no full evaluate_all)."""

def test_condition_b_forecast_includes_sensor_data_in_result():
    from eval_runner import run_condition_b

    out = run_condition_b("Forecast next week's condenser water flow for Chiller 9.")
    assert "sensor_data" in out["result"]
    assert "forecast" in out["result"]
    assert out["result"]["forecast"].get("source") == "mock"


def test_condition_c_disables_knowledge(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from eval_runner import run_condition_c

    out = run_condition_c("Why is Chiller 6 behaving abnormally?")
    assert "metrics" in out
    assert out["metrics"].get("deep_tsfm_invoked") is False


def test_condition_d_runs_without_error(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from eval_runner import run_condition_d

    out = run_condition_d("What sensors are available for Chiller 6?")
    assert "result" in out
    assert out["metrics"].get("deep_tsfm_invoked") is False


def test_theta_sweep_includes_0_95():
    from eval_runner import THETA_VALUES

    assert "0.95" in THETA_VALUES
    # Ensure monotonic order — useful when plotting cost-accuracy curves.
    assert [float(t) for t in THETA_VALUES] == sorted(
        float(t) for t in THETA_VALUES
    )


def test_theta_sweep_straddles_observed_knee():
    """Sweep must include at least one value below and above the ~0.6 knee
    observed in ``eval_results/colab_20260422_1436`` so the frontier plot
    captures both ``deep-off`` and ``deep-on`` regimes."""
    from eval_runner import THETA_VALUES

    floats = [float(t) for t in THETA_VALUES]
    assert any(f <= 0.6 for f in floats), "need a low theta so deep TSFM can skip"
    assert any(f >= 0.9 for f in floats), "need a high theta so deep TSFM fires"


def test_task_completion_fault_diagnosis_with_full_plan():
    from eval_runner import _task_completion_score

    metrics = {
        "plan": [
            "data_retrieval", "anomaly_detection", "root_cause_analysis",
            "validate_failure", "work_order_generation",
        ],
        "skills_skipped": [],
    }
    result = {"work_order": {"id": "WO-1"}, "failure": "vibration_fault"}
    score = _task_completion_score("fault_diagnosis", "Why Chiller 6?", metrics, result)
    assert score == 1.0


def test_task_completion_forecasting_partial_when_artefact_missing():
    from eval_runner import _task_completion_score

    metrics = {"plan": ["data_retrieval", "forecasting"], "skills_skipped": []}
    score = _task_completion_score(
        "forecasting", "Forecast next week", metrics, result={}
    )
    assert 0.6 < score < 0.7


def test_task_completion_condition_a_free_text_answer_credits_keyword():
    from eval_runner import _task_completion_score

    metrics = {"plan": ["direct_llm"], "skills_skipped": []}
    result = {"answer": "You should open a work order for Chiller 6 due to vibration."}
    score = _task_completion_score("fault_diagnosis", "Why?", metrics, result)
    # fault_diagnosis expects 2 artefact keys (work_order, failure); keyword scan
    # only matches "work order" → partial artefact credit (0.5 / 2 = 0.25 wait
    # really 1/2=0.5; then plan_cov=0, exec_cov=0 → (0+0+0.5)/3 ≈ 0.167).
    assert 0.1 < score < 0.25
    # Still strictly better than zero — the free-text answer must get SOME credit.
    assert score > 0.0


def test_task_completion_metadata_full_credit():
    from eval_runner import _task_completion_score

    metrics = {"plan": ["metadata_retrieval"], "skills_skipped": []}
    result = {"metadata": {"sensors": [{"name": "vibration_mm_s"}]}}
    score = _task_completion_score("metadata", "list sensors", metrics, result)
    assert score == 1.0


def test_task_completion_penalizes_skipped_essential_skill():
    from eval_runner import _task_completion_score

    metrics = {
        "plan": [
            "data_retrieval", "anomaly_detection", "root_cause_analysis",
            "validate_failure", "work_order_generation",
        ],
        "skills_skipped": ["work_order_generation"],
    }
    result = {"failure": "vibration_fault"}
    score = _task_completion_score("fault_diagnosis", "diagnose", metrics, result)
    # plan_cov=1.0, exec_cov=0.5, artefact=0.5 → (1+0.5+0.5)/3 ≈ 0.667.
    assert 0.60 < score < 0.70


def test_condition_e_at_theta_0_95_invokes_deep_tsfm(monkeypatch):
    """With θ=0.95, high-severity FMSR (0.92) < θ → deep TSFM should fire,
    charging DEEP_TSFM_COST on top of RCA's static cost."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("RCA_CONFIDENCE_THETA", "0.95")
    monkeypatch.setenv("ENABLE_CONDITIONAL_DEEP_TSFM", "1")
    monkeypatch.setenv("DEEP_TSFM_COST", "1.0")
    from eval_runner import run_condition_e

    out = run_condition_e("Why is Chiller 6 behaving abnormally?")
    assert out["metrics"].get("deep_tsfm_invoked") is True
    # RCA base 0.8 contributes; cost must exceed pre-bump total by ≥ 1.0.
    assert out["metrics"]["total_cost"] >= 0.8 + 1.0
