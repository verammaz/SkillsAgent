import pytest

from pathlib import Path

from scenario_loader import (
    BUILTIN_TASK_BANK,
    infer_tsfm_category,
    load_hf_scenario_tasks,
    load_tsfm_report_tasks,
)


FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "tsfm_report_sample.csv"


def test_builtin_task_bank_len():
    assert len(BUILTIN_TASK_BANK) == 12
    assert all(len(t) == 3 for t in BUILTIN_TASK_BANK)


def test_builtin_task_bank_covers_all_categories():
    categories = {cat for _, _, cat in BUILTIN_TASK_BANK}
    assert categories == {
        "fault_diagnosis",
        "forecasting",
        "anomaly_detection",
        "metadata",
    }


def test_builtin_task_bank_ids_unique():
    ids = [tid for tid, _, _ in BUILTIN_TASK_BANK]
    assert len(set(ids)) == len(ids)


def test_eval_runner_task_bank_is_builtin():
    import eval_runner

    assert eval_runner.TASK_BANK is BUILTIN_TASK_BANK


def test_hf_load_falls_back_when_load_dataset_raises(monkeypatch):
    pytest.importorskip("datasets")
    import datasets

    def boom(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr(datasets, "load_dataset", boom)
    rows = load_hf_scenario_tasks(limit=5)
    assert rows == list(BUILTIN_TASK_BANK)


def test_infer_tsfm_category_tools_and_work_order_override():
    assert infer_tsfm_category({"type": "Workorder", "text": "forecast?", "tsfm_tools_called": "run_tsfm_forecasting"}) == "fault_diagnosis"
    assert infer_tsfm_category({"type": "FMSA", "text": "x", "tsfm_tools_called": "run_integrated_tsad"}) == "fault_diagnosis"
    assert infer_tsfm_category({"type": "multiagent", "text": "What is the forecast?", "tsfm_tools_called": "run_tsfm_forecasting"}) == "forecasting"
    assert infer_tsfm_category({"type": "multiagent", "text": "Any anomalies?", "tsfm_tools_called": "run_integrated_tsad"}) == "anomaly_detection"
    assert infer_tsfm_category({"type": "multiagent", "text": "Anomalies and should I create a work order?", "tsfm_tools_called": "run_integrated_tsad"}) == "fault_diagnosis"


def test_load_tsfm_report_fixture():
    rows = load_tsfm_report_tasks(FIXTURE_CSV)
    assert len(rows) == 3
    assert rows[0][0] == "TSFM_501"
    assert rows[0][2] == "anomaly_detection"
    assert rows[1][2] == "forecasting"
    assert rows[2][2] == "fault_diagnosis"
