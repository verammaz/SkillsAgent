# tools.py
#
# Thin wrappers around the four AssetOpsBench agents:
#   IoT agent   → get_sensor_data, get_asset_metadata (uv → servers.iot.main)
#   TSFM agent  → forecast_sensor (run_tsfm_forecasting subprocess when context allows);
#                  deep_tsfm_refine_anomalies (run_integrated_tsad subprocess);
#                  detect_anomaly is local profile + IQR heuristics (no TSFM subprocess).
#   FMSR agent  → map_failure (uv → servers.fmsr.main)
#   WO agent    → generate_work_order (uv → servers.wo.tools; equipment map chillers 6/9)
#
# Set ASSETOPS to the AssetOpsBench ``src`` directory.  IoT needs
# CouchDB up and ``.env`` in the AssetOpsBench repo root.  Use
# ``USE_IOT_SUBPROCESS=1`` / ``USE_WO_SUBPROCESS=1`` / ``USE_TSFM_SUBPROCESS=1``
# to enable (defaults on when ASSETOPS is set for TSFM as well).
#
# TSFM calls ``servers.tsfm.main`` (``run_tsfm_forecasting``, ``run_integrated_tsad``,
# ``fetch_tsfm_catalog`` reads ``servers.tsfm.models`` static lists).
# Requires ``tsfm_public``, model checkpoints, and PATH_TO_* vars in AssetOpsBench
# ``.env``. For official TSFM scenario CSVs, set ``PATH_TO_DATASETS_DIR`` (see
# ``tsfm_task_spec``) or place files under ``<repo>/data/tsfm_test_data/``.
# Falls back to lightweight heuristics when ML stack is unavailable.

import csv
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from tsfm_task_spec import parse_official_tsfm_forecast_task, resolve_tsfm_dataset_path


# ── AssetOpsBench subprocess helper (shared with FMSR) ───────────────────────

def _assetops_repo_root() -> Path | None:
    """Parent of ``src`` — the AssetOpsBench repo root for ``uv run``."""
    src = os.getenv("ASSETOPS")
    if not src:
        return None
    return Path(src).resolve().parent


def _uv_run_assetops_json(script: str, *, timeout: int = 120) -> dict:
    """Run ``uv run python -c`` in the AssetOpsBench repo; stdout must be one JSON object.

    Uses ``--no-sync`` so uv does not re-resolve the lockfile on every call — the
    AssetOpsBench lockfile pulls ``huggingface-hub>=1.0`` via ``litellm``/``openai-agents``,
    but ``transformers 4.57`` (required by ``granite-tsfm``) demands ``<1.0``. We
    install the correct hf-hub imperatively during setup and then freeze the venv.
    """
    root = _assetops_repo_root()
    if not root:
        raise RuntimeError("ASSETOPS is not set — point it at AssetOpsBench/src.")

    proc = subprocess.run(
        ["uv", "run", "--no-sync", "python", "-c", script],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or (proc.stdout or "").strip()
        raise RuntimeError(f"uv run failed: {err}")

    out = proc.stdout.strip()
    if not out:
        raise RuntimeError("uv run returned empty stdout")
    return json.loads(out)


def _canonical_chiller_name(asset_id: str) -> str:
    """``Chiller 6`` / ``chiller_6`` → ``Chiller 6`` (CouchDB asset_id)."""
    m = re.search(r"chiller\s*(\d+)", asset_id.lower())
    if m:
        return f"Chiller {m.group(1)}"
    return asset_id.strip() or "Chiller 6"


def _pivot_observations_to_readings(observations: list) -> dict[str, list[float]]:
    """Turn IoT ``history`` observations into per-sensor series (time-ordered)."""
    if not observations:
        return {}

    def ts_key(doc: dict) -> str:
        return doc.get("timestamp") or ""

    sorted_docs = sorted(observations, key=ts_key)
    series: dict[str, list[float]] = {}

    for doc in sorted_docs:
        for k, v in doc.items():
            if k in ("_id", "_rev", "asset_id", "timestamp"):
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                series.setdefault(k, []).append(float(v))

    return series


def _iot_history_window(lookback_days: int) -> tuple[str, str | None]:
    """ISO range for CouchDB query.

    Default window is June~2020 so the bundled ``chiller`` DB fixture returns rows.
    Override with ``IOT_HISTORY_START`` / ``IOT_HISTORY_FINAL`` (ISO strings).
    """
    start_env = os.getenv("IOT_HISTORY_START")
    final_env = os.getenv("IOT_HISTORY_FINAL")
    if start_env:
        return start_env, final_env

    start_dt = datetime(2020, 6, 1, 0, 0, 0)
    final_dt = start_dt + timedelta(days=max(1, lookback_days))
    return start_dt.isoformat(), final_dt.isoformat()


def _get_sensor_data_subprocess(asset_id: str, lookback_days: int) -> dict | None:
    """Fetch history via ``servers.iot.main.history`` in AssetOpsBench venv."""
    site = os.getenv("IOT_SITE_NAME", "MAIN")
    canonical = _canonical_chiller_name(asset_id)
    start, final = _iot_history_window(lookback_days)
    final_py = "None" if final is None else repr(final)

    script = f"""
import json, sys
sys.path.insert(0, "src")
from servers.iot.main import history
site = {json.dumps(site)}
asset_id = {json.dumps(canonical)}
start = {json.dumps(start)}
final = {final_py}
r = history(site, asset_id, start, final)
out = r.model_dump() if hasattr(r, "model_dump") else r.dict()
print(json.dumps(out))
"""
    try:
        data = _uv_run_assetops_json(script, timeout=90)
    except Exception:
        return None

    if data.get("error"):
        return None

    observations = data.get("observations") or []
    readings = _pivot_observations_to_readings(observations)
    if not readings:
        return None

    return {
        "asset_id": asset_id,
        "lookback_days": lookback_days,
        "readings": readings,
        "iot_site": site,
        "iot_canonical_asset": canonical,
        "iot_history_start": start,
        "iot_history_final": final,
        "iot_total_observations": data.get("total_observations", len(observations)),
        "source": "iot_subprocess",
    }


def _get_asset_metadata_subprocess(asset_id: str) -> dict | None:
    site = os.getenv("IOT_SITE_NAME", "MAIN")
    canonical = _canonical_chiller_name(asset_id)
    script = f"""
import json, sys
sys.path.insert(0, "src")
from servers.iot.main import sensors
site = {json.dumps(site)}
asset_id = {json.dumps(canonical)}
r = sensors(site, asset_id)
out = r.model_dump() if hasattr(r, "model_dump") else r.dict()
print(json.dumps(out))
"""
    try:
        data = _uv_run_assetops_json(script, timeout=60)
    except Exception:
        return None

    if data.get("error"):
        return None

    names = data.get("sensors") or []
    rows = [{"name": n, "unit": "", "description": ""} for n in names]
    return {
        "asset_id": asset_id,
        "sensors": rows,
        "source": "iot_subprocess",
    }


# Equipment IDs used by WO sample data (see AssetOpsBench scenarios)
_CHILLER_EQUIPMENT_ID = {"6": "CWC04006", "9": "CWC04009"}


def _equipment_id_for_chiller(asset_id: str) -> str | None:
    m = re.search(r"chiller\s*(\d+)", asset_id.lower())
    if not m:
        return None
    return _CHILLER_EQUIPMENT_ID.get(m.group(1))


# Filenames of the bundled WO datasets in
# ``AssetOpsBench/src/tmp/assetopsbench/sample_data/``.
# These are what ``servers.wo.data.load(dataset)`` would otherwise fetch from CouchDB.
_WO_DATASET_FILES = {
    "wo_events":             "all_wo_with_code_component_events.csv",
    "primary_failure_codes": "primary_failure_codes.csv",
    "failure_codes":         "failure_codes.csv",
    "events":                "event.csv",
    "alert_events":          "alert_events.csv",
}


def _generate_work_order_subprocess(asset_id: str, failure: str, priority: str) -> dict | None:
    """Use ``predict_next_work_order`` + structured template.

    When ``WO_CSV_DIR`` is set (or the default AssetOpsBench sample_data path exists),
    we monkeypatch ``servers.wo.tools.load`` inside the subprocess so WO runs against
    local CSVs and doesn't need CouchDB.
    """
    eq = _equipment_id_for_chiller(asset_id)
    if not eq:
        return None

    csv_dir = os.getenv("WO_CSV_DIR")
    if not csv_dir:
        root = _assetops_repo_root()
        if root:
            default = root / "src" / "tmp" / "assetopsbench" / "sample_data"
            if (default / "all_wo_with_code_component_events.csv").is_file():
                csv_dir = str(default)

    script = f"""
import json, os, sys
sys.path.insert(0, "src")

_csv_dir = {json.dumps(csv_dir) if csv_dir else "None"}
if _csv_dir:
    import pandas as pd
    from servers.wo import data as _wo_data
    import servers.wo.tools as _wo_tools

    _DS_FILE = {json.dumps(_WO_DATASET_FILES)}
    _cache = {{}}

    def _load_from_csv(dataset):
        if dataset in _cache:
            return _cache[dataset]
        path = os.path.join(_csv_dir, _DS_FILE.get(dataset, dataset + ".csv"))
        if not os.path.isfile(path):
            _cache[dataset] = None
            return None
        df = pd.read_csv(path)
        for col in _wo_data._DATE_COLS.get(dataset, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        _cache[dataset] = df
        return df

    _wo_tools.load = _load_from_csv
    _wo_data.load  = _load_from_csv

from servers.wo.tools import predict_next_work_order
r = predict_next_work_order({json.dumps(eq)}, None, None)
out = r.model_dump() if hasattr(r, "model_dump") else r.dict()
print(json.dumps(out))
"""
    try:
        data = _uv_run_assetops_json(script, timeout=60)
    except Exception:
        return None

    if data.get("error"):
        return None

    preds = data.get("predictions") or []
    top = preds[0] if preds else {}
    wo_id = f"WO-PRED-{eq}-{abs(hash(failure)) % 9000 + 1000}"
    return {
        "work_order_id": wo_id,
        "asset_id": asset_id,
        "equipment_id": eq,
        "failure": failure,
        "priority": priority,
        "description": (
            f"Address {failure} on {asset_id}. "
            f"Next likely WO category: {top.get('primary_code_description', top.get('primary_code', 'see history'))}."
        ),
        "predicted_next_categories": preds[:5],
        "source": "wo_subprocess",
        "steps": [
            "Review predicted next work-order types from historical transitions",
            f"Inspect equipment {eq} for {failure}",
            "Schedule corrective or preventive action per site policy",
        ],
        "estimated_downtime_hours": 4.0,
    }


def _use_iot_subprocess() -> bool:
    if not _assetops_repo_root():
        return False
    return os.getenv("USE_IOT_SUBPROCESS", "1").lower() in ("1", "true", "yes")


def _use_wo_subprocess() -> bool:
    if not _assetops_repo_root():
        return False
    return os.getenv("USE_WO_SUBPROCESS", "1").lower() in ("1", "true", "yes")


def _use_tsfm_subprocess() -> bool:
    if not _assetops_repo_root():
        return False
    return os.getenv("USE_TSFM_SUBPROCESS", "1").lower() in ("1", "true", "yes")


def fetch_tsfm_catalog() -> dict:
    """Static TSFM task and model lists — same data as TSFM MCP ``get_ai_tasks`` / ``get_tsfm_models``.

    Imports ``servers.tsfm.models`` (the MCP tools wrap these lists). We avoid
    ``servers.tsfm.main`` here because it pulls ``mcp``/FastMCP, which may not be
    installed in every SkillsAgent environment.

    Requires ``servers`` importable (typically ``PYTHONPATH`` includes AssetOpsBench
    ``src``).

    Returns:
        ``{"ai_tasks": [...], "models": [...]}`` with dict-shaped entries, or
        ``{"ai_tasks": [], "models": [], "error": "<reason>"}`` on failure.
    """
    try:
        from servers.tsfm import models as tsfm_models
    except ImportError as exc:
        return {"ai_tasks": [], "models": [], "error": f"import: {exc}"}

    try:
        tasks = getattr(tsfm_models, "_AI_TASKS", []) or []
        models = getattr(tsfm_models, "_TSFM_MODELS", []) or []
        return {
            "ai_tasks": [dict(t) for t in tasks],
            "models": [dict(m) for m in models],
        }
    except Exception as exc:
        return {"ai_tasks": [], "models": [], "error": str(exc)}


def _tsfm_effective_horizon(horizon_days: int) -> int:
    cap = int(os.getenv("TSFM_MAX_FORECAST_STEPS", "28"))
    return min(max(int(horizon_days), 1), cap)


def _write_tsfm_series_csv(
    readings: dict, column: str, path: str, *, min_rows: int
) -> bool:
    raw = readings.get(column)
    if not isinstance(raw, list) or len(raw) < min_rows:
        return False
    vals = [float(x) for x in raw]
    base = datetime(2020, 6, 1, 0, 0, 0)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", column])
        for i, v in enumerate(vals):
            ts = base + timedelta(hours=i)
            w.writerow([ts.isoformat(), v])
    return True


def _tsfm_forecast_subprocess(
    abs_csv: str,
    target_column: str,
    horizon: int,
    *,
    timestamp_column: str = "timestamp",
    conditional_columns: list[str] | None = None,
) -> dict | None:
    root = _assetops_repo_root()
    if not root:
        return None
    dp = json.dumps(abs_csv)
    tc = json.dumps(target_column)
    ts_col = json.dumps(timestamp_column)
    mc = json.dumps(os.getenv("TSFM_MODEL_CHECKPOINT", "ttm_96_28"))
    hsteps = int(horizon)
    cond_py = repr(conditional_columns)
    script = f"""
import json, sys
sys.path.insert(0, "src")
from servers.tsfm.main import run_tsfm_forecasting
import numpy as np
r = run_tsfm_forecasting(
    dataset_path={dp},
    timestamp_column={ts_col},
    target_columns=[{tc}],
    model_checkpoint={mc},
    forecast_horizon={hsteps},
    conditional_columns={cond_py},
    frequency_sampling="oov",
    autoregressive_modeling=True,
)
d = r.model_dump() if hasattr(r, "model_dump") else r.dict()
if d.get("error"):
    print(json.dumps({{"error": d["error"]}}))
    sys.exit(0)
path = d.get("results_file")
if not path:
    print(json.dumps({{"error": "no results_file"}}))
    sys.exit(0)
with open(path) as pred_fh:
    pred = json.load(pred_fh)
tp = pred.get("target_prediction", [])
if not tp:
    print(json.dumps({{"error": "empty prediction"}}))
    sys.exit(0)
a = np.asarray(tp, dtype=float)
lim = min({hsteps}, a.shape[1] if a.ndim >= 2 else len(a))
if a.ndim >= 3:
    row = a[0, :lim, 0]
elif a.ndim == 2:
    row = a[0, :lim]
else:
    row = a[:lim]
series = [float(x) for x in np.atleast_1d(row).flatten()]
print(json.dumps({{"forecasted": series}}))
"""
    try:
        proc = subprocess.run(
            ["uv", "run", "--no-sync", "python", "-c", script],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=int(os.getenv("TSFM_SUBPROCESS_TIMEOUT", "300")),
            env=os.environ.copy(),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    line = (proc.stdout or "").strip()
    if not line:
        return None
    try:
        out = json.loads(line.splitlines()[-1])
    except json.JSONDecodeError:
        return None
    if out.get("error"):
        return None
    fc = out.get("forecasted")
    if not fc:
        return None
    return {"forecasted": fc}


def _deep_tsfm_primary_column(anomaly: dict, readings: dict) -> str | None:
    for d in anomaly.get("anomaly_details") or []:
        if not isinstance(d, str):
            continue
        head = d.split(":", 1)[0].strip()
        if head in readings:
            return head
    if not readings:
        return None
    return max(
        readings.keys(),
        key=lambda k: len(readings[k]) if isinstance(readings[k], list) else 0,
    )


def _tsfm_integrated_tsad_subprocess(abs_csv: str, target_column: str) -> tuple[int, list[str]] | None:
    root = _assetops_repo_root()
    if not root:
        return None
    dp = json.dumps(abs_csv)
    tc = json.dumps(target_column)
    mc = json.dumps(os.getenv("TSFM_MODEL_CHECKPOINT", "ttm_96_28"))
    cn = repr(target_column)
    script = f"""
import csv, json, sys
sys.path.insert(0, "src")

# NumPy 2.0 compatibility shim: AssetOpsBench TSAD references np.infty,
# which was removed in NumPy 2.0. Restore the alias before any tsfm import.
import numpy as _np
if not hasattr(_np, "infty"):
    _np.infty = _np.inf
if not hasattr(_np, "Inf"):
    _np.Inf = _np.inf

# AssetOpsBench bug workaround: anomaly._get_tsfm_dataloaders calls
# ForecastDFDataset(**column_specifiers, ..., id_columns=id_columns) but
# forecasting.py injects id_columns into column_specifiers, which Python
# then rejects as "got multiple values for keyword argument 'id_columns'".
# We strip id_columns from column_specifiers before the spread.
from servers.tsfm import anomaly as _A
_orig_get_loaders = _A._get_tsfm_dataloaders
def _patched_get_loaders(df_dataframe, model_config, dataset_config_dictionary, scaling=False):
    cs = dataset_config_dictionary.get("column_specifiers", {{}})
    if isinstance(cs, dict) and "id_columns" in cs:
        fixed_cfg = dict(dataset_config_dictionary)
        fixed_cfg["column_specifiers"] = {{k: v for k, v in cs.items() if k != "id_columns"}}
        dataset_config_dictionary = fixed_cfg
    return _orig_get_loaders(df_dataframe, model_config, dataset_config_dictionary, scaling=scaling)
_A._get_tsfm_dataloaders = _patched_get_loaders
# main.py may have already imported the name; rebind there too.
try:
    from servers.tsfm import main as _M
    if hasattr(_M, "_get_tsfm_dataloaders"):
        _M._get_tsfm_dataloaders = _patched_get_loaders
except Exception:
    pass

from servers.tsfm.main import run_integrated_tsad
r = run_integrated_tsad(
    dataset_path={dp},
    timestamp_column="timestamp",
    target_columns=[{tc}],
    model_checkpoint={mc},
    frequency_sampling="oov",
    autoregressive_modeling=True,
)
d = r.model_dump() if hasattr(r, "model_dump") else r.dict()
if d.get("error"):
    print(json.dumps({{"error": d["error"]}}))
    sys.exit(0)
path = d.get("results_file")
if not path:
    print(json.dumps({{"error": "no results_file"}}))
    sys.exit(0)
count = 0
with open(path, newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        lab = row.get("anomaly_label", "")
        try:
            if float(lab) != 0.0:
                count += 1
        except (TypeError, ValueError):
            if str(lab).lower() in ("true", "1", "yes"):
                count += 1
msgs = []
if count:
    msgs.append(
        "TSFM integrated TSAD: "
        + str(count)
        + " conformal anomaly record(s) on "
        + {cn}
    )
print(json.dumps({{"anomaly_records": count, "messages": msgs}}))
"""
    try:
        proc = subprocess.run(
            ["uv", "run", "--no-sync", "python", "-c", script],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=int(os.getenv("TSFM_SUBPROCESS_TIMEOUT", "300")),
            env=os.environ.copy(),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    line = (proc.stdout or "").strip()
    if not line:
        return None
    try:
        out = json.loads(line.splitlines()[-1])
    except json.JSONDecodeError:
        return None
    if out.get("error"):
        return None
    return (int(out.get("anomaly_records", 0)), list(out.get("messages") or []))


# ── IoT agent ────────────────────────────────────────────────────────────────

def _get_sensor_data_from_csv(asset_id: str, lookback_days: int) -> dict | None:
    """Read ``{IOT_CSV_DIR}/{asset_key}.csv`` (produced by ``scripts/extract_main_json.py``).

    Used on environments without a live CouchDB (e.g. Colab). The CSV must have a
    ``timestamp`` header plus one column per sensor. We return the last
    ``min(all_rows, IOT_CSV_MAX_ROWS or 2000)`` rows so TSFM has enough context.
    """
    csv_dir = os.getenv("IOT_CSV_DIR")
    if not csv_dir:
        return None
    key = _normalize_asset(asset_id)
    path = Path(csv_dir) / f"{key}.csv"
    if not path.is_file():
        return None

    max_rows = int(os.getenv("IOT_CSV_MAX_ROWS", "2000"))

    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header or header[0] != "timestamp":
            return None
        sensor_cols = header[1:]
        readings: dict[str, list[float]] = {c: [] for c in sensor_cols}
        timestamps: list[str] = []
        for row in reader:
            if len(row) != len(header):
                continue
            timestamps.append(row[0])
            for c, v in zip(sensor_cols, row[1:]):
                if v == "" or v is None:
                    continue
                try:
                    readings[c].append(float(v))
                except ValueError:
                    pass

    if max_rows and len(timestamps) > max_rows:
        timestamps = timestamps[-max_rows:]
        for c in sensor_cols:
            if len(readings[c]) > max_rows:
                readings[c] = readings[c][-max_rows:]

    # drop empty columns so they don't confuse downstream skills
    readings = {c: v for c, v in readings.items() if v}
    if not readings:
        return None

    return {
        "asset_id": asset_id,
        "lookback_days": lookback_days,
        "readings": readings,
        "iot_history_start": timestamps[0] if timestamps else None,
        "iot_history_final": timestamps[-1] if timestamps else None,
        "iot_total_observations": len(timestamps),
        "source": "iot_csv",
    }


def get_sensor_data(asset_id: str, lookback_days: int = 7) -> dict:
    """Fetch time-series sensor readings for an asset.

    Resolution order (first hit wins):
      1. ``IOT_CSV_DIR`` CSV — Colab-friendly, bypasses CouchDB.
      2. AssetOpsBench ``servers.iot.main`` subprocess (needs CouchDB).
      3. Hardcoded 5-row mock — last resort for unit tests / offline dev.
    """
    csv_data = _get_sensor_data_from_csv(asset_id, lookback_days)
    if csv_data is not None:
        return csv_data

    if _use_iot_subprocess():
        real = _get_sensor_data_subprocess(asset_id, lookback_days)
        if real is not None:
            return real

    return {
        "asset_id": asset_id,
        "lookback_days": lookback_days,
        "readings": {
            "vibration_mm_s": [1.2, 1.4, 3.8, 4.1, 5.2],
            "CHW_supply_temp_F": [44.1, 44.3, 44.0, 47.2, 48.5],
            "compressor_power_kW": [210, 212, 215, 230, 265],
            "flow_rate_GPM": [980, 975, 940, 890, 820],
        },
        "source": "mock",
    }


def get_asset_metadata(asset_id: str) -> dict:
    """Fetch sensor catalog and unit descriptions for an asset."""
    csv_dir = os.getenv("IOT_CSV_DIR")
    if csv_dir:
        key = _normalize_asset(asset_id)
        path = Path(csv_dir) / f"{key}.csv"
        if path.is_file():
            with open(path, "r", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader, None)
            if header and header[0] == "timestamp" and len(header) > 1:
                rows = [
                    {"name": c, "unit": "", "description": ""} for c in header[1:]
                ]
                return {"asset_id": asset_id, "sensors": rows, "source": "iot_csv"}

    if _use_iot_subprocess():
        real = _get_asset_metadata_subprocess(asset_id)
        if real is not None:
            return real

    catalog = {
        "chiller_6": {
            "sensors": [
                {"name": "vibration_mm_s", "unit": "mm/s", "description": "Compressor vibration"},
                {"name": "CHW_supply_temp_F", "unit": "°F", "description": "Chilled water supply temperature"},
                {"name": "CHW_return_temp_F", "unit": "°F", "description": "Chilled water return temperature"},
                {"name": "compressor_power_kW", "unit": "kW", "description": "Compressor electrical draw"},
                {"name": "flow_rate_GPM", "unit": "GPM", "description": "Chilled water flow rate"},
            ],
        },
        "chiller_9": {
            "sensors": [
                {"name": "condenser_flow_GPM", "unit": "GPM", "description": "Condenser water flow rate"},
                {"name": "compressor_speed_rpm", "unit": "RPM", "description": "Compressor shaft speed"},
                {"name": "refrigerant_psi", "unit": "PSI", "description": "Refrigerant pressure"},
                {"name": "COP", "unit": "", "description": "Coefficient of performance"},
            ],
        },
    }
    key = _normalize_asset(asset_id)
    return {"asset_id": asset_id, "sensors": catalog.get(key, {}).get("sensors", []), "source": "mock"}


# ── TSFM agent ───────────────────────────────────────────────────────────────

# Default limits keyed by *profile* id (short names). Mock readings use these keys
# directly; CouchDB / IoT uses long labels — see ``_IOT_SENSOR_PROFILE_RULES``.
SENSOR_PROFILE_LIMITS: dict[str, dict] = {
    "vibration_mm_s": {"max": 4.5},
    "CHW_supply_temp_F": {"max": 46.0},
    "CHW_return_temp_F": {"max": 58.0},
    "condenser_return_to_tower_temp_F": {"max": 99.0, "min": 55.0},
    "evaporator_liquid_temp_F": {"max": 50.0, "min": 28.0},
    "setpoint_temp_F": {"max": 52.0, "min": 36.0},
    "condenser_water_flow": {"min": 3500.0},
    "compressor_power_kW": {"max": 250.0},
    "power_input_kw": {"max": 2000.0, "min": 5.0},
    "flow_rate_GPM": {"min": 850.0},
    "tonnage": {"min": 50.0, "max": 6000.0},
    "chiller_efficiency": {"max": 0.95, "min": 0.05},
    "pct_loaded": {"max": 100.0, "min": 0.0},
}

# Order matters: more specific patterns first (e.g. tower return before generic return).
_IOT_SENSOR_PROFILE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"condenser\s+water\s+return\s+to\s+tower\s+temperature",
            re.I,
        ),
        "condenser_return_to_tower_temp_F",
    ),
    (
        re.compile(
            r"liquid\s+refrigerant\s+evaporator\s+temperature",
            re.I,
        ),
        "evaporator_liquid_temp_F",
    ),
    (re.compile(r"setpoint\s+temperature", re.I), "setpoint_temp_F"),
    (re.compile(r"supply\s+temperature", re.I), "CHW_supply_temp_F"),
    (re.compile(r"return\s+temperature", re.I), "CHW_return_temp_F"),
    (re.compile(r"condenser\s+water\s+flow", re.I), "condenser_water_flow"),
    (re.compile(r"power\s+input", re.I), "power_input_kw"),
    (re.compile(r"chiller\s+%\s+loaded|%\s+loaded", re.I), "pct_loaded"),
    (re.compile(r"chiller\s+efficiency|efficiency", re.I), "chiller_efficiency"),
    (re.compile(r"\btonnage\b", re.I), "tonnage"),
    (re.compile(r"vibration", re.I), "vibration_mm_s"),
)


def _map_sensor_name_to_profile(sensor: str) -> str | None:
    """Map a reading column (short mock key or long IoT label) to a profile id."""
    if sensor in SENSOR_PROFILE_LIMITS:
        return sensor
    for rx, profile in _IOT_SENSOR_PROFILE_RULES:
        if rx.search(sensor):
            return profile
    return None


# Planner / knowledge often use short sensor ids; IoT columns are long labels.
_FORECAST_HINT_TO_PROFILE: dict[str, str] = {
    "flow_rate_GPM": "condenser_water_flow",
    "condenser_flow_GPM": "condenser_water_flow",
    "CHW_supply_temp_F": "CHW_supply_temp_F",
    "CHW_return_temp_F": "CHW_return_temp_F",
    "vibration_mm_s": "vibration_mm_s",
    "compressor_power_kW": "compressor_power_kW",
}


def _find_reading_key_for_forecast(sensor_hint: str, readings: dict) -> str | None:
    if sensor_hint in readings:
        return sensor_hint
    prof = _FORECAST_HINT_TO_PROFILE.get(sensor_hint)
    if prof is None:
        prof = _map_sensor_name_to_profile(sensor_hint)
    if prof:
        for k in readings:
            if _map_sensor_name_to_profile(k) == prof:
                return k
    h = sensor_hint.lower().replace("_", " ")
    for k in readings:
        kl = k.lower()
        if h in kl or sensor_hint.lower() in kl:
            return k
    return None


def _resolve_sensor_limits(sensor: str, overrides: dict | None) -> dict:
    """Resolve min/max limits: per-sensor overrides, then profile defaults."""
    ovr = overrides if isinstance(overrides, dict) else {}
    if sensor in ovr:
        return dict(ovr[sensor])
    profile = _map_sensor_name_to_profile(sensor)
    if profile is None:
        return {}
    base = dict(SENSOR_PROFILE_LIMITS.get(profile, {}))
    if profile in ovr and isinstance(ovr[profile], dict):
        base.update(ovr[profile])
    return base


def _iqr_tail_message(sensor: str, values: list, mult: float) -> str | None:
    """Flag last point if it is far from the series median in IQR units."""
    vals = [float(x) for x in values]
    if len(vals) < 4:
        return None
    sorted_v = sorted(vals)
    n = len(sorted_v)
    q1 = sorted_v[n // 4]
    q3 = sorted_v[(3 * n) // 4]
    iqr = q3 - q1
    if abs(iqr) < 1e-12:
        return None
    med = sorted_v[n // 2]
    last = vals[-1]
    if abs(last - med) > mult * iqr:
        return f"{sensor}: last value {last:.4g} far from median {med:.4g} (>{mult}×IQR)"
    return None


def detect_anomaly(data: dict, thresholds: dict = None, *, iqr_mult: float | None = None) -> dict:
    """Run lightweight anomaly detection on sensor readings.

    Resolves min/max from ``SENSOR_PROFILE_LIMITS`` using short keys (mock data)
    or long CouchDB-style column names (see ``_IOT_SENSOR_PROFILE_RULES``).
    Optional ``thresholds`` overrides per exact column name or per profile id
    (e.g. ``CHW_supply_temp_F``).

    If no profile matches and no limits apply, uses a robust IQR-vs-last-value
    check. ``iqr_mult`` tightens that band (e.g. deep TSFM stand-in uses ~1.0).
    """
    readings = data.get("readings", {}) if isinstance(data, dict) else {}
    ovr = thresholds if isinstance(thresholds, dict) else {}
    mult = iqr_mult if iqr_mult is not None else float(os.getenv("ANOMALY_IQR_MULT", "1.5"))

    anomalies = []
    for sensor, values in readings.items():
        vals = values if isinstance(values, list) else [values]
        limits = _resolve_sensor_limits(sensor, ovr)

        if limits.get("max") is not None and any(v > limits["max"] for v in vals):
            anomalies.append(f"{sensor} exceeded max {limits['max']}")
        if limits.get("min") is not None and any(v < limits["min"] for v in vals):
            anomalies.append(f"{sensor} fell below min {limits['min']}")
        if not limits:
            msg = _iqr_tail_message(sensor, vals, mult)
            if msg:
                anomalies.append(msg)

    severity = "high" if len(anomalies) >= 2 else ("medium" if anomalies else "none")
    return {
        "anomalies_detected": bool(anomalies),
        "anomaly_details": anomalies,
        "severity": severity,
    }


def forecast_sensor(
    asset_id: str,
    sensor: str,
    horizon_days: int = 7,
    sensor_data: dict | None = None,
    task: str | None = None,
) -> dict:
    """Predict future sensor values over ``horizon_days`` (step cap via TSFM_MAX_FORECAST_STEPS).

    When the natural-language ``task`` matches an official AssetOpsBench TSFM inference
    prompt and the referenced CSV exists (``PATH_TO_DATASETS_DIR`` or
    ``<AssetOpsBench>/data/tsfm_test_data/``), runs ``run_tsfm_forecasting`` on that file
    with the parsed timestamp and target column names.

    Otherwise, when ``sensor_data`` has a long enough series and ``USE_TSFM_SUBPROCESS``
    is on, builds a temporary single-series CSV from IoT readings. If that fails,
    uses a lightweight mock.
    """
    readings = (sensor_data or {}).get("readings") or {}
    h = _tsfm_effective_horizon(horizon_days)

    if task and _use_tsfm_subprocess():
        spec = parse_official_tsfm_forecast_task(task)
        if spec is not None:
            bench_csv = resolve_tsfm_dataset_path(
                spec.dataset_ref, assetops_repo_root=_assetops_repo_root()
            )
            if bench_csv is not None:
                cond = list(spec.conditional_columns) if spec.conditional_columns else None
                real = _tsfm_forecast_subprocess(
                    str(bench_csv.resolve()),
                    spec.target_column,
                    h,
                    timestamp_column=spec.timestamp_column,
                    conditional_columns=cond,
                )
                if real and real.get("forecasted"):
                    fc = real["forecasted"]
                    trend = "increasing" if fc[-1] > fc[0] else "stable"
                    return {
                        "asset_id": asset_id,
                        "sensor": sensor,
                        "resolved_column": spec.target_column,
                        "horizon_days": horizon_days,
                        "forecast_steps": len(fc),
                        "forecasted": fc,
                        "trend": trend,
                        "source": "tsfm_subprocess_bench_csv",
                        "dataset_path": str(bench_csv),
                        "timestamp_column": spec.timestamp_column,
                    }

    if _use_tsfm_subprocess() and readings:
        col = _find_reading_key_for_forecast(sensor, readings)
        if col:
            min_rows = int(os.getenv("TSFM_MIN_CONTEXT_ROWS", "96"))
            fd, csv_path = tempfile.mkstemp(suffix=".csv")
            try:
                os.close(fd)
                if _write_tsfm_series_csv(readings, col, csv_path, min_rows=min_rows):
                    real = _tsfm_forecast_subprocess(
                        str(Path(csv_path).resolve()), col, h
                    )
                    if real and real.get("forecasted"):
                        fc = real["forecasted"]
                        trend = "increasing" if fc[-1] > fc[0] else "stable"
                        return {
                            "asset_id": asset_id,
                            "sensor": sensor,
                            "resolved_column": col,
                            "horizon_days": horizon_days,
                            "forecast_steps": len(fc),
                            "forecasted": fc,
                            "trend": trend,
                            "source": "tsfm_subprocess",
                        }
            finally:
                try:
                    os.unlink(csv_path)
                except OSError:
                    pass

    import random

    base = {"flow_rate_GPM": 900, "vibration_mm_s": 2.0, "compressor_power_kW": 220}
    base_val = base.get(sensor, 100)
    forecasted = [round(base_val + random.uniform(-10, 20), 2) for _ in range(h)]
    return {
        "asset_id": asset_id,
        "sensor": sensor,
        "horizon_days": horizon_days,
        "forecasted": forecasted,
        "trend": "increasing" if forecasted[-1] > forecasted[0] else "stable",
        "source": "mock",
    }


# ── FMSR agent ───────────────────────────────────────────────────────────────

def _fmsr_get_failure_modes(asset_id: str) -> list[str]:
    """Fetch real failure modes from the FMSR MCP server via uv subprocess.

    Calls the AssetOpsBench uv environment directly — no mcp package required
    in this env.  Returns a list of failure mode description strings, e.g.:
      ["Compressor Overheating: Failed due to Normal wear, overheating", ...]

    Raises RuntimeError if ASSETOPS is not set or the subprocess fails.
    """
    assetops = os.getenv("ASSETOPS")
    if not assetops:
        raise RuntimeError("ASSETOPS env var not set — cannot reach FMSR server.")

    repo_root = str(Path(assetops).parent)
    # "Chiller 6" → "chiller",  "chiller_9" → "chiller"
    asset_name = re.sub(r"[\s_]*\d+", "", asset_id).strip().lower()

    script = (
        "import sys, json; "
        "sys.path.insert(0, 'src'); "
        "from servers.fmsr.main import get_failure_modes; "
        f"r = get_failure_modes('{asset_name}'); "
        "print(json.dumps(r.model_dump()))"
    )
    proc = subprocess.run(
        ["uv", "run", "--no-sync", "python", "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FMSR subprocess error: {proc.stderr.strip()}")

    data = json.loads(proc.stdout.strip())
    return data.get("failure_modes", [])


def _fmsr_short_name(fmsr_mode: str) -> str:
    """Compact canonical name from a verbose FMSR description.

    "Compressor Overheating: Failed due to wear"  →  "compressor_overheating"
    "Evaporator Water side fouling"               →  "evaporator_water_side_fouling"
    """
    name = fmsr_mode.split(":")[0].strip()
    return re.sub(r"\s+", "_", name).lower()


def map_failure_with_meta(
    anomaly: dict, failure_modes: list = None, asset_id: str = None
) -> dict:
    """Map anomalies to a failure mode; returns dict for confidence gating.

    Keys: ``failure`` (str), ``matched_via`` (``"fmsr"`` | ``"knowledge"`` | ``"unknown"``).
    """
    details = (
        " ".join(anomaly.get("anomaly_details", [])).lower()
        if isinstance(anomaly, dict)
        else ""
    )

    if asset_id:
        try:
            fmsr_modes = _fmsr_get_failure_modes(asset_id)
            for fm in fmsr_modes:
                keywords = re.findall(r"[a-z]{4,}", fm.lower())
                if any(kw in details for kw in keywords):
                    return {
                        "failure": _fmsr_short_name(fm),
                        "matched_via": "fmsr",
                    }
        except Exception:
            pass

    if not failure_modes:
        failure_modes = [
            {"name": "cavitation", "symptoms": ["vibration", "flow"]},
            {"name": "bearing_failure", "symptoms": ["vibration", "power"]},
            {"name": "condenser_fouling", "symptoms": ["temp", "power"]},
            {"name": "refrigerant_leak", "symptoms": ["pressure", "temp", "COP"]},
        ]
    for mode in failure_modes:
        if any(kw in details for kw in mode.get("symptoms", [])):
            return {"failure": mode["name"], "matched_via": "knowledge"}
    return {"failure": "unknown_failure", "matched_via": "unknown"}


# Failure-mode lexicon keyed to substrings that typically appear in the user's
# task prompt. Used by ``_task_specificity_score`` to reward tasks that name the
# symptom / subsystem the FMSR diagnosis identified (proposal Sec.~\\ref{sec:conf}).
_FAILURE_KEYWORDS = {
    "vibration":              ("vibration", "vibrat", "shake"),
    "temperature":            ("temperature", "temp", "overheat", "cooling", "chw", "chilled"),
    "flow":                   ("flow", "gpm", "pump"),
    "compressor":             ("compressor", "motor", "bearing"),
    "power":                  ("power", "kw", "electrical", "draw", "load"),
    "refrigerant":            ("refrigerant", "psi", "pressure", "leak"),
    "condenser":              ("condenser", "tower"),
    "evaporator":             ("evaporator",),
    "efficiency":             ("efficiency", "cop", "performance"),
    "setpoint":               ("setpoint", "set point", "target"),
}


def _task_specificity_score(task: str | None, failure: str) -> float:
    """Return 0–1 reflecting how well the task's wording matches the diagnosed failure.

    A vague prompt (``"Why is Chiller 6 behaving abnormally?"``) scores ~0.3 — the
    FMSR diagnosis is plausible but the user gave us no corroborating symptom.
    A specific prompt that names the subsystem the diagnosis implicates
    (``"Chiller 9 vibration has been rising"`` → ``vibration_fault``) scores 1.0.
    """
    if not task or not isinstance(task, str):
        return 0.3
    t = task.lower()
    fail_lower = (failure or "").lower()

    matched_buckets = 0
    diagnosis_matched = False
    for bucket, kws in _FAILURE_KEYWORDS.items():
        if any(k in t for k in kws):
            matched_buckets += 1
            if bucket in fail_lower:
                diagnosis_matched = True

    if diagnosis_matched:
        return 1.0
    if matched_buckets >= 2:
        return 0.7
    if matched_buckets == 1:
        return 0.55
    return 0.3


def score_diagnosis_confidence(
    anomaly: dict,
    meta: dict,
    *,
    sensor_data: dict | None = None,
    wo_history: list | None = None,
    task: str | None = None,
) -> float:
    """Graded FMSR diagnosis confidence in [0, 1] (proposal Fig.~2, Alg.~2).

    Combines six signals so θ sweeps produce a continuous curve rather than a
    step at a single bucket edge:

      1. Match quality (``matched_via``): fmsr > knowledge > unknown; heavy
         penalty when ``failure == "unknown_failure"``.
      2. Severity + density: severity bucket + log-scaled anomaly count.
      3. Sensor coverage: fraction of monitored sensors flagged, peaked at ~30%
         (too few → spurious, too many → systemic confusion).
      4. TSAD corroboration: non-zero integrated-TSAD conformal records boost
         confidence (so deep TSFM raises the second-pass score).
      5. WO history support: small bonus if a past work order mentions the
         diagnosed failure for this asset.
      6. Task specificity: does the user's prompt name a subsystem that is
         consistent with the diagnosed failure? Vague prompts get a neutral-low
         score, specific prompts that corroborate the diagnosis score 1.0.

    Signals (3)–(6) degrade gracefully when inputs are missing, so the legacy
    two-arg call still works for back-compat. Output clipped to [0.05, 0.98].
    """
    import math

    failure = meta.get("failure", "unknown_failure") if isinstance(meta, dict) else "unknown_failure"
    via = meta.get("matched_via", "unknown") if isinstance(meta, dict) else "unknown"
    sev = (anomaly or {}).get("severity", "none")
    details = (anomaly or {}).get("anomaly_details") or []

    # 1. match-quality ∈ [0, 1]
    via_score = {"fmsr": 0.90, "knowledge": 0.60}.get(via, 0.15)
    if failure in ("unknown_failure", None, ""):
        via_score *= 0.25  # no failure mode identified

    # 2. severity + density ∈ [0, 1]
    sev_bucket = {"high": 0.70, "medium": 0.50, "low": 0.35, "none": 0.10}.get(sev, 0.25)
    density_bonus = min(0.30, 0.09 * math.log1p(len(details)))
    sev_score = min(1.0, sev_bucket + density_bonus)

    # 3. coverage ∈ [0, 1], peaked at ~30% of monitored sensors
    coverage_score = 0.5  # neutral prior when we can't compute it
    readings = (sensor_data or {}).get("readings") if isinstance(sensor_data, dict) else None
    if isinstance(readings, dict) and readings:
        affected = set()
        for d in details:
            for name in readings.keys():
                if name and name in d:
                    affected.add(name)
                    break
        frac = len(affected) / max(1, len(readings))
        coverage_score = max(0.2, 1.0 - abs(frac - 0.3) * 1.5)

    # 4. TSAD corroboration ∈ [0, 1] — populated after deep TSFM
    tsad_records = int((anomaly or {}).get("tsfm_integrated_tsad_records") or 0)
    tsad_score = min(1.0, tsad_records / 20.0)  # saturates at 20 records

    # 5. WO history support ∈ {0, 1}
    wo_score = 0.0
    if wo_history and failure not in ("unknown_failure", None, ""):
        fail_lower = str(failure).lower().replace("_", " ")
        for w in wo_history:
            if fail_lower in str(w).lower():
                wo_score = 1.0
                break

    # 6. Task specificity ∈ [0, 1]
    task_spec_score = _task_specificity_score(task, failure)

    # Weighted combination (weights sum to 1.0)
    conf = (
        0.30 * via_score
        + 0.22 * sev_score
        + 0.18 * coverage_score
        + 0.13 * tsad_score
        + 0.05 * wo_score
        + 0.12 * task_spec_score
    )
    return round(max(0.05, min(0.98, conf)), 3)


def deep_tsfm_refine_anomalies(
    asset_id: str,
    sensor_data: dict,
    anomaly: dict,
    anomaly_definition: dict | None = None,
) -> dict:
    """Deep pass: optional integrated TSAD subprocess plus stricter IQR/profile check.

    When ``USE_TSFM_SUBPROCESS`` is on and readings are long enough, runs
    ``run_integrated_tsad`` on a primary column; merges messages with the
    existing tight-threshold ``detect_anomaly`` path. ``anomaly_definition``
    comes from the Anomaly Definition knowledge plugin (``min_context_rows``, etc.);
    env ``TSFM_MIN_CONTEXT_ROWS`` overrides when the plugin is absent.

    ``asset_id`` is reserved for future routing; subprocess uses repo env (``ASSETOPS``).
    """
    _ = asset_id
    ad = anomaly_definition if isinstance(anomaly_definition, dict) else {}
    readings = sensor_data.get("readings") or {} if isinstance(sensor_data, dict) else {}
    tsad_msgs: list[str] = []
    tsad_count = 0
    csv_path: str | None = None

    if "min_context_rows" in ad:
        min_rows = int(ad["min_context_rows"])
    else:
        min_rows = int(os.getenv("TSFM_MIN_CONTEXT_ROWS", "96"))

    if _use_tsfm_subprocess() and readings:
        col = _deep_tsfm_primary_column(anomaly, readings)
        if col:
            fd, csv_path = tempfile.mkstemp(suffix=".csv")
            try:
                os.close(fd)
                if _write_tsfm_series_csv(readings, col, csv_path, min_rows=min_rows):
                    got = _tsfm_integrated_tsad_subprocess(
                        str(Path(csv_path).resolve()), col
                    )
                    if got:
                        tsad_count, tsad_msgs = got[0], got[1]
            finally:
                if csv_path:
                    try:
                        os.unlink(csv_path)
                    except OSError:
                        pass

    tight: dict[str, dict] = {}
    for k, lim in SENSOR_PROFILE_LIMITS.items():
        row: dict = {}
        if "max" in lim:
            row["max"] = lim["max"] * 0.92
        if "min" in lim:
            row["min"] = lim["min"] * 1.05
        if row:
            tight[k] = row

    refined = detect_anomaly(sensor_data, tight, iqr_mult=1.0)
    merged_details = list(
        dict.fromkeys(
            (anomaly.get("anomaly_details") or [])
            + (refined.get("anomaly_details") or [])
            + tsad_msgs
        )
    )
    merged = {
        "anomalies_detected": bool(
            refined.get("anomalies_detected")
            or anomaly.get("anomalies_detected", False)
            or bool(tsad_msgs)
            or tsad_count > 0
        ),
        "anomaly_details": merged_details,
        "severity": refined.get("severity") or anomaly.get("severity", "none"),
        "deep_tsfm_refined": True,
    }
    if merged["anomalies_detected"] and merged["severity"] == "none":
        merged["severity"] = "medium"
    if tsad_count > 0:
        merged["tsfm_integrated_tsad_records"] = tsad_count
    return merged


def map_failure(anomaly: dict, failure_modes: list = None, asset_id: str = None) -> str:
    """Map anomaly patterns to the most likely known failure mode.

    When asset_id is provided, queries the real FMSR MCP server first.
    Falls back to the provided failure_modes list or hardcoded defaults.
    """
    return map_failure_with_meta(anomaly, failure_modes, asset_id)["failure"]


# ── WO agent ─────────────────────────────────────────────────────────────────

def generate_work_order(asset_id: str, failure: str, priority: str = "high") -> dict:
    """Create a structured maintenance work order."""
    if _use_wo_subprocess():
        real = _generate_work_order_subprocess(asset_id, failure, priority)
        if real is not None:
            return real

    wo_id = f"WO-{asset_id.replace(' ', '').upper()}-{abs(hash(failure)) % 9000 + 1000}"
    return {
        "work_order_id": wo_id,
        "asset_id": asset_id,
        "failure": failure,
        "priority": priority,
        "description": f"Inspect and repair {failure} on {asset_id}.",
        "steps": [
            "Isolate and safely shut down the chiller",
            f"Inspect components associated with {failure}",
            "Replace or repair the faulty component",
            "Perform post-repair diagnostics",
            "Return chiller to service and monitor for 24 h",
        ],
        "estimated_downtime_hours": 4.0,
        "source": "mock",
    }


# ── shared helper ─────────────────────────────────────────────────────────────

def _normalize_asset(asset_id: str) -> str:
    m = re.search(r"chiller\s*(\d+)", asset_id.lower())
    return f"chiller_{m.group(1)}" if m else asset_id.lower().replace(" ", "_")
