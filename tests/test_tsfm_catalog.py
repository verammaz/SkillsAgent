"""TSFM static catalog (AssetOps ``servers.tsfm`` parity)."""

import pytest


def test_fetch_tsfm_catalog_matches_server_lists():
    from tools import fetch_tsfm_catalog

    out = fetch_tsfm_catalog()
    if out.get("error"):
        pytest.skip(f"servers.tsfm unavailable in this environment: {out['error']}")

    assert not out.get("error")
    tasks = out["ai_tasks"]
    models = out["models"]
    assert len(tasks) >= 4
    assert len(models) >= 4
    ids = {t["task_id"] for t in tasks}
    assert "tsfm_forecasting" in ids
    assert any(m["model_id"] == "ttm_96_28" for m in models)
