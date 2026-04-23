import pytest

from scenario_loader import BUILTIN_TASK_BANK, load_hf_scenario_tasks


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
