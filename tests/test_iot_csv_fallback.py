"""Unit tests for IOT_CSV_DIR fallback in tools.get_sensor_data/get_asset_metadata."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path


def _write_csv(path: Path, n_rows: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "flow_rate_GPM", "vibration_mm_s"])
        for i in range(n_rows):
            w.writerow([f"2022-01-01T{i:02d}:00:00", 900 + i, 2.0 + i * 0.01])


def _import_tools():
    import tools

    return importlib.reload(tools)


def test_csv_fallback_short_circuits_before_subprocess(tmp_path, monkeypatch):
    _write_csv(tmp_path / "chiller_6.csv", n_rows=200)
    monkeypatch.setenv("IOT_CSV_DIR", str(tmp_path))
    monkeypatch.setenv("USE_IOT_SUBPROCESS", "1")  # would normally try CouchDB
    monkeypatch.delenv("ASSETOPS", raising=False)

    tools = _import_tools()
    d = tools.get_sensor_data("Chiller 6", lookback_days=7)

    assert d["source"] == "iot_csv"
    assert d["iot_total_observations"] == 200
    assert set(d["readings"]) == {"flow_rate_GPM", "vibration_mm_s"}
    assert len(d["readings"]["flow_rate_GPM"]) == 200


def test_csv_fallback_respects_max_rows(tmp_path, monkeypatch):
    _write_csv(tmp_path / "chiller_9.csv", n_rows=2500)
    monkeypatch.setenv("IOT_CSV_DIR", str(tmp_path))
    monkeypatch.setenv("IOT_CSV_MAX_ROWS", "500")

    tools = _import_tools()
    d = tools.get_sensor_data("Chiller 9", lookback_days=14)

    assert d["source"] == "iot_csv"
    assert d["iot_total_observations"] == 500
    assert len(d["readings"]["flow_rate_GPM"]) == 500


def test_metadata_from_csv(tmp_path, monkeypatch):
    _write_csv(tmp_path / "chiller_6.csv", n_rows=5)
    monkeypatch.setenv("IOT_CSV_DIR", str(tmp_path))

    tools = _import_tools()
    m = tools.get_asset_metadata("Chiller 6")

    assert m["source"] == "iot_csv"
    names = {s["name"] for s in m["sensors"]}
    assert names == {"flow_rate_GPM", "vibration_mm_s"}


def test_missing_csv_falls_through_to_mock(tmp_path, monkeypatch):
    # empty dir — no chiller_*.csv
    monkeypatch.setenv("IOT_CSV_DIR", str(tmp_path))
    monkeypatch.setenv("USE_IOT_SUBPROCESS", "0")  # also no subprocess
    monkeypatch.delenv("ASSETOPS", raising=False)

    tools = _import_tools()
    d = tools.get_sensor_data("Chiller 6")

    assert d["source"] == "mock"
