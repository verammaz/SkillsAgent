"""Stream a CouchDB ``main.json`` export into per-asset CSVs for real TSFM runs.

The export is typically ~1 GB with one doc per (asset, timestamp). We stream
with ``ijson``, keep only the asset(s) requested, sort by timestamp, and write::

    <outdir>/<asset_key>.csv      # timestamp,<sensor1>,<sensor2>,...

Usage::

    python scripts/extract_main_json.py \\
        --input /path/to/main.json \\
        --outdir data/chillers \\
        --assets "Chiller 6,Chiller 9" \\
        --max-rows 5000

Runs a single pass over the file (~90s-3min depending on disk/RAM).
Requires ``ijson`` — install with ``pip install ijson`` if missing.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def _asset_key(asset_id: str) -> str:
    m = re.search(r"chiller\s*(\d+)", asset_id.lower())
    return f"chiller_{m.group(1)}" if m else asset_id.lower().replace(" ", "_")


def extract(
    *,
    input_path: Path,
    outdir: Path,
    assets: list[str] | None,
    max_rows: int | None,
    drop_all_zero: bool,
) -> dict[str, int]:
    try:
        import ijson  # type: ignore
    except ImportError:
        sys.exit("ijson is required: pip install ijson")

    outdir.mkdir(parents=True, exist_ok=True)
    targets = {a.strip() for a in assets} if assets else None

    # rows[asset_key] = list[tuple[timestamp_str, {col: val}]]
    rows: dict[str, list[tuple[str, dict]]] = {}
    cols: dict[str, list[str]] = {}  # preserve first-seen column order

    with open(input_path, "rb") as f:
        for doc in ijson.items(f, "docs.item"):
            if not isinstance(doc, dict):
                continue
            aid = doc.get("asset_id")
            ts = doc.get("timestamp")
            if not aid or not ts:
                continue
            if targets is not None and aid not in targets:
                continue

            key = _asset_key(str(aid))
            sensor_vals = {
                k: v
                for k, v in doc.items()
                if k not in {"_id", "_rev", "asset_id", "timestamp"}
                and not k.startswith("_")
            }
            if drop_all_zero and all(
                v in (0, 0.0, None, "") for v in sensor_vals.values()
            ):
                continue

            seen = cols.setdefault(key, [])
            for c in sensor_vals:
                if c not in seen:
                    seen.append(c)
            rows.setdefault(key, []).append((str(ts), sensor_vals))

    written: dict[str, int] = {}
    for key, recs in rows.items():
        recs.sort(key=lambda r: r[0])
        if max_rows and len(recs) > max_rows:
            recs = recs[-max_rows:]

        col_list = cols[key]
        out_path = outdir / f"{key}.csv"
        with open(out_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["timestamp", *col_list])
            for ts, vals in recs:
                w.writerow([ts, *(vals.get(c, "") for c in col_list)])
        written[key] = len(recs)
        print(f"  wrote {out_path}  ({len(recs)} rows × {len(col_list)} sensors)")

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="Path to main.json")
    ap.add_argument(
        "--outdir", required=True, type=Path, help="Output directory for CSVs"
    )
    ap.add_argument(
        "--assets",
        default="",
        help="Comma-separated asset_ids to keep (e.g. 'Chiller 6,Chiller 9'). Empty = all.",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Keep only the last N rows per asset (default: all).",
    )
    ap.add_argument(
        "--keep-zero-rows",
        action="store_true",
        help="Include rows where every sensor value is 0/empty (off by default).",
    )
    ns = ap.parse_args()

    assets = [a for a in ns.assets.split(",") if a.strip()] if ns.assets else None
    out = extract(
        input_path=ns.input,
        outdir=ns.outdir,
        assets=assets,
        max_rows=ns.max_rows,
        drop_all_zero=not ns.keep_zero_rows,
    )
    print(f"done: {sum(out.values())} rows across {len(out)} asset(s)")


if __name__ == "__main__":
    main()
