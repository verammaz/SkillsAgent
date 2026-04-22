"""Smoke test: verify IoT subprocess path produces real readings for Chiller 6.

Run from the SkillsAgent directory:
    python scripts/smoke_iot.py

Expect: source == "iot_subprocess" (or "iot_csv" if IOT_CSV_DIR is set) and
non-empty readings for Chiller 6.
"""

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    import dotenv

    dotenv.load_dotenv(REPO / ".env")
except ImportError:
    env_path = REPO / ".env"
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

from tools import get_asset_metadata, get_sensor_data  # noqa: E402


def main() -> None:
    asset = "Chiller 6"
    if not os.getenv("ASSETOPS") and not os.getenv("IOT_CSV_DIR"):
        print("ERROR: Set ASSETOPS or IOT_CSV_DIR in .env")
        raise SystemExit(1)

    data = get_sensor_data(asset, lookback_days=7)
    meta = get_asset_metadata(asset)

    readings = data.get("readings") or {}
    preview = {
        k: (v[:5] if isinstance(v, list) and len(v) > 5 else v)
        for k, v in list(readings.items())[:4]
    }
    summary = {
        "source": data.get("source"),
        "iot_total_observations": data.get("iot_total_observations"),
        "iot_history_start": data.get("iot_history_start"),
        "iot_history_final": data.get("iot_history_final"),
        "num_sensor_series": len(readings),
        "readings_preview": preview,
        "metadata_source": meta.get("source"),
        "metadata_sensor_count": len(meta.get("sensors") or []),
    }
    print(json.dumps(summary, indent=2))

    if data.get("source") == "mock":
        print("\nWARNING: Falling back to mock. Check ASSETOPS/IOT_CSV_DIR and CouchDB.")
        raise SystemExit(2)
    if not readings:
        print("\nERROR: readings empty — query window may not match loaded data.")
        raise SystemExit(3)

    print(f"\nOK: IoT path is live (source={data.get('source')}).")


if __name__ == "__main__":
    main()
