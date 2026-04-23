import json
from pathlib import Path

from couch_export_catalog import build_sensor_catalog_from_export


def test_build_catalog_from_tiny_export(tmp_path: Path):
    p = tmp_path / "export.json"
    p.write_text(
        json.dumps(
            {
                "docs": [
                    {
                        "_id": "1",
                        "asset_id": "Chiller 6",
                        "timestamp": "2022-01-01T00:00:00Z",
                        "Chiller 6 Supply Temperature": 44.0,
                        "Chiller 6 Condenser Water Flow": 4000.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cat = build_sensor_catalog_from_export(p, max_total_docs=5)
    assert "chiller_6" in cat
    assert "Chiller 6 Supply Temperature" in cat["chiller_6"]["sensors"]


def test_sensor_metadata_plugin_merges_export(monkeypatch, tmp_path):
    p = tmp_path / "export.json"
    p.write_text(
        json.dumps(
            {
                "docs": [
                    {
                        "asset_id": "Chiller 6",
                        "timestamp": "t",
                        "Chiller 6 X Sensor": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("COUCHDB_EXPORT_PATH", str(p))
    from knowledge import SensorMetadataPlugin

    pl = SensorMetadataPlugin()
    out = pl.retrieve("metadata_retrieval", "Chiller 6 question", {})
    sensors = out["sensor_metadata"].get("sensors") or []
    assert "Chiller 6 X Sensor" in sensors
