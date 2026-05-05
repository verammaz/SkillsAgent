"""Fetch and snapshot scenarios from AssetOpsBench scenario server.

This creates a reproducible scenario file that can be reused by eval_runner
without refetching from the server.

Example:
  python scripts/export_scenarios.py \
      --scenario-set 13aab653-66fe-4fe6-84d8-89f1b18eede3 \
      --output eval_inputs/tsfm_set/scenarios.jsonl
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _http_json(url: str) -> dict:
    req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}: {msg}") from exc
    except urllib.error.URLError as exc:
        hint = ""
        if "localhost" in url or "127.0.0.1" in url:
            hint = (
                " If you are on Google Colab, localhost is the Colab VM, not your laptop. "
                "Use ngrok (or similar) and pass --server-url with that HTTPS URL."
            )
        raise RuntimeError(f"Cannot reach scenario server {url}: {exc.reason!s}.{hint}") from exc


def export_scenarios(
    *,
    server_url: str,
    scenario_set_id: str,
    output_path: Path,
    limit: int | None,
) -> tuple[int, str]:
    url = f"{server_url.rstrip('/')}/scenario-set/{scenario_set_id}"
    payload = _http_json(url)
    title = str(payload.get("title") or "")
    scenarios = payload.get("scenarios", []) or []
    if limit is not None:
        scenarios = scenarios[: max(0, limit)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with open(output_path, "w", encoding="utf-8") as fh:
        for i, s in enumerate(scenarios):
            row = {
                "scenario_id": str(s.get("id", f"S_{i}")),
                "query": str(s.get("query", "")),
                "metadata": s.get("metadata", {}) or {},
                "scenario_set_id": scenario_set_id,
                "scenario_set_title": title,
                "fetched_at": ts,
            }
            fh.write(json.dumps(row, default=str) + "\n")
    return len(scenarios), title


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server-url", default="http://localhost:8099")
    ap.add_argument("--scenario-set", required=True, help="Scenario-set UUID from /scenario-types.")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    ap.add_argument("--limit", type=int, default=None, help="Optional max rows.")
    ns = ap.parse_args()

    n, title = export_scenarios(
        server_url=ns.server_url,
        scenario_set_id=ns.scenario_set,
        output_path=ns.output,
        limit=ns.limit,
    )
    print(f"exported {n} scenarios  title={title!r}  -> {ns.output}")


if __name__ == "__main__":
    main()

