"""Tests for scripts/grade_assetops_metrics helpers (no uv / ReActXen required)."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SKILLS = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "grade_assetops_metrics",
    _SKILLS / "scripts" / "grade_assetops_metrics.py",
)
assert _SPEC and _SPEC.loader
_gam = importlib.util.module_from_spec(_SPEC)
sys.modules["grade_assetops_metrics"] = _gam
_SPEC.loader.exec_module(_gam)


def test_load_rubric_sample(tmp_path: Path):
    _load_rubric = _gam._load_rubric

    p = tmp_path / "r.json"
    p.write_text(
        json.dumps(
            [
                {"id": 201, "type": "TSFM", "text": "Q?", "characteristic_form": "C1"},
                {"id": 999, "type": "OTHER", "text": "x", "characteristic_form": "y"},
            ]
        ),
        encoding="utf-8",
    )
    r = _load_rubric(p)
    assert r["201"]["text"] == "Q?"
    assert r["201"]["characteristic_form"] == "C1"
    assert "999" not in r


def test_build_payload_row(tmp_path: Path):
    _build_payload = _gam._build_payload

    rubric = {
        "201": {"text": "Q?", "characteristic_form": "expected"},
    }
    csv_path = tmp_path / "ab.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "task_id",
                "condition",
                "theta",
                "result_json",
                "trace_json",
                "error",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "task_id": "201",
                "condition": "A",
                "theta": "",
                "result_json": json.dumps({"answer": "hello"}),
                "trace_json": json.dumps({"plan": ["x"]}),
                "error": "",
            }
        )
    pl = _build_payload(csv_path, rubric)
    assert len(pl) == 1
    assert pl[0]["result"] == "hello"
    assert "plan" in pl[0]["trace"]


def test_aggregate_coerces_csv_strings():
    _aggregate_by_condition = _gam._aggregate_by_condition

    graded = [
        {
            "condition": "A",
            "theta": "",
            "overall_correct": "False",
            "task_completion": "True",
            "hallucinations": "True",
        },
        {
            "condition": "A",
            "theta": "",
            "overall_correct": "true",
            "task_completion": "false",
            "hallucinations": "false",
        },
    ]
    agg = _aggregate_by_condition(graded)
    assert len(agg) == 1
    assert agg[0]["n"] == 2
    assert agg[0]["overall_correct_rate"] == 0.5
    assert agg[0]["task_completion_rate"] == 0.5
    assert agg[0]["hallucinations_rate"] == 0.5
