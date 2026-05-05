"""Unit tests for official TSFM scenario parsing and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from tsfm_task_spec import parse_official_tsfm_forecast_task, resolve_tsfm_dataset_path


def test_parse_scenario_216():
    q = (
        "Forecast 'Chiller 9 Condenser Water Flow' using data in "
        "'chiller9_annotated_small_test.csv'. Use parameter 'Timestamp' as a timestamp. "
        "Use the following parameters as inputs 'Chiller 9 Liquid Refrigerant Evaporator Temperature,"
        "Chiller 9 Return Temperature'"
    )
    s = parse_official_tsfm_forecast_task(q)
    assert s is not None
    assert s.dataset_ref == "chiller9_annotated_small_test.csv"
    assert s.target_column == "Chiller 9 Condenser Water Flow"
    assert s.timestamp_column == "Timestamp"
    assert "Chiller 9 Return Temperature" in s.conditional_columns


def test_parse_scenario_218():
    q = (
        "Use data in 'chiller9_annotated_small_test.csv' to forecast "
        "'Chiller 9 Condenser Water Flow' with 'Timestamp' as a timestamp."
    )
    s = parse_official_tsfm_forecast_task(q)
    assert s is not None
    assert s.dataset_ref == "chiller9_annotated_small_test.csv"
    assert s.target_column == "Chiller 9 Condenser Water Flow"
    assert s.timestamp_column == "Timestamp"
    assert s.conditional_columns == ()


def test_parse_unquoted_csv_anomaly_task_returns_none():
    q = (
        "I need to perform Time Series anomaly detection of 'Chiller 9 Condenser Water Flow' "
        "using data in chiller9_tsad.csv"
    )
    assert parse_official_tsfm_forecast_task(q) is None


def test_resolve_prefers_path_to_datasets_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    csv_path = tmp_path / "chiller9_annotated_small_test.csv"
    csv_path.write_text("a,b\n1,2\n")
    monkeypatch.setenv("PATH_TO_DATASETS_DIR", str(tmp_path))
    out = resolve_tsfm_dataset_path(
        "chiller9_annotated_small_test.csv",
        assetops_repo_root=None,
    )
    assert out == csv_path.resolve()


def test_resolve_under_data_tsfm_test_data(tmp_path: Path):
    repo = tmp_path / "AssetOpsBench"
    data_dir = repo / "data" / "tsfm_test_data"
    data_dir.mkdir(parents=True)
    f = data_dir / "chiller9_annotated_small_test.csv"
    f.write_text("x\n")
    out = resolve_tsfm_dataset_path(
        "chiller9_annotated_small_test.csv",
        assetops_repo_root=repo,
    )
    assert out == f.resolve()
