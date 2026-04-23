"""End-to-end check that WO runs against local CSVs (no CouchDB).

Skipped when AssetOpsBench/uv aren't available so the test suite is still
runnable on bare dev machines.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

import pytest


def _assetops_root() -> Path | None:
    src = os.getenv("ASSETOPS")
    return Path(src).resolve().parent if src else None


@pytest.mark.skipif(
    shutil.which("uv") is None,
    reason="uv not installed",
)
@pytest.mark.skipif(
    _assetops_root() is None,
    reason="ASSETOPS env var not set",
)
@pytest.mark.skipif(
    _assetops_root() is not None
    and not (_assetops_root() / "src" / "tmp" / "assetopsbench" / "sample_data" / "all_wo_with_code_component_events.csv").is_file(),
    reason="AssetOpsBench WO sample_data not present",
)
def test_generate_work_order_uses_local_csv(monkeypatch):
    monkeypatch.setenv("USE_WO_SUBPROCESS", "1")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import tools

    tools = importlib.reload(tools)
    r = tools.generate_work_order("Chiller 6", "compressor_overheating", priority="high")

    assert r["source"] == "wo_subprocess", f"expected wo_subprocess, got {r['source']}"
    assert r["equipment_id"] == "CWC04006"
    cats = r.get("predicted_next_categories") or []
    assert cats, "no predictions returned"
    # Every prediction must carry a real Markov probability and a primary_code
    for c in cats:
        assert "primary_code" in c and c["primary_code"]
        assert 0.0 <= float(c.get("probability", -1)) <= 1.0
