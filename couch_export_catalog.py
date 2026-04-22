"""Extract sensor *names* from a CouchDB JSON export (e.g. ``main.json``) without loading all docs.

Typical export shape::

    {"docs": [ {"_id": "...", "asset_id": "Chiller 6", "timestamp": "...", "Chiller 6 Supply Temperature": 42.0, ...}, ... ]}

The file can contain millions of rows — we stream the first ``max_total_docs`` with
``ijson`` when available, else parse the first document from a prefix of the file.

Environment:

- ``COUCHDB_EXPORT_PATH`` — if set, :class:`SensorMetadataPlugin` merges live columns
  into knowledge (Phase F).

Optional dependency: ``ijson`` for large exports; without it, only the first ~256 KiB
is scanned for one JSON object.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SKIP_KEYS = frozenset({"_id", "_rev", "asset_id", "timestamp"})


def _columns_from_doc(doc: dict) -> list[str]:
    return [k for k in doc if k not in _SKIP_KEYS and not k.startswith("_")]


def _first_doc_from_prefix(path: Path, max_bytes: int = 262144) -> dict | None:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        buf = f.read(max_bytes)
    m = re.search(r'"docs"\s*:\s*\[\s*\{', buf)
    if not m:
        return None
    start = m.end() - 1
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(buf)):
        c = buf[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(buf[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_asset_key(asset_id: str) -> str:
    m = re.search(r"chiller\s*(\d+)", asset_id.lower())
    return f"chiller_{m.group(1)}" if m else asset_id.lower().replace(" ", "_")


def build_sensor_catalog_from_export(
    path: str | Path,
    *,
    max_total_docs: int = 24,
) -> dict[str, dict[str, Any]]:
    """Return ``{ "chiller_6": {"sensors": [...], "source": "couch_export"}, ... }``."""
    path = Path(path)
    if not path.is_file():
        logger.warning("CouchDB export not found: %s", path)
        return {}

    columns_by_asset: dict[str, set[str]] = defaultdict(set)
    seen = 0

    try:
        import ijson

        with open(path, "rb") as f:
            for doc in ijson.items(f, "docs.item"):
                if not isinstance(doc, dict):
                    continue
                aid = doc.get("asset_id")
                if not aid:
                    continue
                key = _normalize_asset_key(str(aid))
                for col in _columns_from_doc(doc):
                    columns_by_asset[key].add(col)
                seen += 1
                if seen >= max_total_docs:
                    break
    except ImportError:
        doc = _first_doc_from_prefix(path)
        if doc:
            aid = doc.get("asset_id")
            if aid:
                key = _normalize_asset_key(str(aid))
                for col in _columns_from_doc(doc):
                    columns_by_asset[key].add(col)
        else:
            logger.warning(
                "Install ``ijson`` for streaming large CouchDB exports; prefix parse failed."
            )
    except Exception as e:
        logger.warning("Couch export scan failed: %s", e)
        doc = _first_doc_from_prefix(path)
        if doc and doc.get("asset_id"):
            key = _normalize_asset_key(str(doc["asset_id"]))
            for col in _columns_from_doc(doc):
                columns_by_asset[key].add(col)

    out: dict[str, dict[str, Any]] = {}
    for asset_key, cols in columns_by_asset.items():
        out[asset_key] = {
            "sensors": sorted(cols),
            "source": "couch_export",
        }
    return out
