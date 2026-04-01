# tools.py
#
# Thin wrappers around the four AssetOpsBench agents:
#   IoT agent   → get_sensor_data, get_asset_metadata
#   TSFM agent  → detect_anomaly, forecast_sensor
#   FMSR agent  → map_failure
#   WO agent    → generate_work_order
#A
# Every function currently returns mock data so the agent runs offline.
# Replace each function body with the real AssetOpsBench agent call when ready:
#
#   from assetopsbench.agents import IoTAgent
#   return IoTAgent().query(asset_id, lookback_days=lookback_days)

import re


# ── IoT agent ────────────────────────────────────────────────────────────────

def get_sensor_data(asset_id: str, lookback_days: int = 7) -> dict:
    """Fetch time-series sensor readings for an asset."""
    # TODO: replace with real IoT agent call
    return {
        "asset_id":     asset_id,
        "lookback_days": lookback_days,
        "readings": {
            "vibration_mm_s":      [1.2, 1.4, 3.8, 4.1, 5.2],
            "CHW_supply_temp_F":   [44.1, 44.3, 44.0, 47.2, 48.5],
            "compressor_power_kW": [210, 212, 215, 230, 265],
            "flow_rate_GPM":       [980, 975, 940, 890, 820],
        },
    }


def get_asset_metadata(asset_id: str) -> dict:
    """Fetch sensor catalog and unit descriptions for an asset."""
    # TODO: replace with real IoT agent call
    catalog = {
        "chiller_6": {
            "sensors": [
                {"name": "vibration_mm_s",      "unit": "mm/s", "description": "Compressor vibration"},
                {"name": "CHW_supply_temp_F",   "unit": "°F",   "description": "Chilled water supply temperature"},
                {"name": "CHW_return_temp_F",   "unit": "°F",   "description": "Chilled water return temperature"},
                {"name": "compressor_power_kW", "unit": "kW",   "description": "Compressor electrical draw"},
                {"name": "flow_rate_GPM",        "unit": "GPM",  "description": "Chilled water flow rate"},
            ],
        },
        "chiller_9": {
            "sensors": [
                {"name": "condenser_flow_GPM",   "unit": "GPM", "description": "Condenser water flow rate"},
                {"name": "compressor_speed_rpm", "unit": "RPM", "description": "Compressor shaft speed"},
                {"name": "refrigerant_psi",      "unit": "PSI", "description": "Refrigerant pressure"},
                {"name": "COP",                  "unit": "",    "description": "Coefficient of performance"},
            ],
        },
    }
    key = _normalize_asset(asset_id)
    return {"asset_id": asset_id, "sensors": catalog.get(key, {}).get("sensors", [])}


# ── TSFM agent ───────────────────────────────────────────────────────────────

def detect_anomaly(data: dict, thresholds: dict = None) -> dict:
    """Run anomaly detection on sensor readings."""
    # TODO: replace with real TSFM agent call
    readings = data.get("readings", {}) if isinstance(data, dict) else {}
    default_thresholds = {
        "vibration_mm_s":      {"max": 4.5},
        "CHW_supply_temp_F":   {"max": 46.0},
        "compressor_power_kW": {"max": 250.0},
        "flow_rate_GPM":       {"min": 850.0},
    }
    thresholds = thresholds or default_thresholds
    anomalies = []
    for sensor, values in readings.items():
        limits = thresholds.get(sensor, {})
        vals = values if isinstance(values, list) else [values]
        if limits.get("max") and any(v > limits["max"] for v in vals):
            anomalies.append(f"{sensor} exceeded max {limits['max']}")
        if limits.get("min") and any(v < limits["min"] for v in vals):
            anomalies.append(f"{sensor} fell below min {limits['min']}")

    severity = "high" if len(anomalies) >= 2 else ("medium" if anomalies else "none")
    return {
        "anomalies_detected": bool(anomalies),
        "anomaly_details":    anomalies,
        "severity":           severity,
    }


def forecast_sensor(asset_id: str, sensor: str, horizon_days: int = 7) -> dict:
    """Predict future sensor values over horizon_days."""
    try:
        from assetopsbench.agents import TSFMAgent
    except ImportError as exc:
        raise RuntimeError("assetopsbench with TSFMAgent is required for forecasting") from exc

    return TSFMAgent().query(asset_id, sensor=sensor, horizon_days=horizon_days)


# ── FMSR agent ───────────────────────────────────────────────────────────────

def map_failure(anomaly: dict, failure_modes: list = None) -> str:
    """Map anomaly patterns to the most likely known failure mode."""
    # TODO: replace with real FMSR agent call
    if not failure_modes:
        failure_modes = [
            {"name": "cavitation",        "symptoms": ["vibration", "flow"]},
            {"name": "bearing_failure",   "symptoms": ["vibration", "power"]},
            {"name": "condenser_fouling", "symptoms": ["temp", "power"]},
            {"name": "refrigerant_leak",  "symptoms": ["pressure", "temp", "COP"]},
        ]
    details = " ".join(anomaly.get("anomaly_details", [])).lower() if isinstance(anomaly, dict) else ""
    for mode in failure_modes:
        if any(kw in details for kw in mode.get("symptoms", [])):
            return mode["name"]
    return "unknown_failure"


# ── WO agent ─────────────────────────────────────────────────────────────────

def generate_work_order(asset_id: str, failure: str, priority: str = "high") -> dict:
    """Create a structured maintenance work order."""
    # TODO: replace with real WO agent call
    wo_id = f"WO-{asset_id.replace(' ', '').upper()}-{abs(hash(failure)) % 9000 + 1000}"
    return {
        "work_order_id":            wo_id,
        "asset_id":                 asset_id,
        "failure":                  failure,
        "priority":                 priority,
        "description":              f"Inspect and repair {failure} on {asset_id}.",
        "steps": [
            "Isolate and safely shut down the chiller",
            f"Inspect components associated with {failure}",
            "Replace or repair the faulty component",
            "Perform post-repair diagnostics",
            "Return chiller to service and monitor for 24 h",
        ],
        "estimated_downtime_hours": 4.0,
    }


# ── shared helper ─────────────────────────────────────────────────────────────

def _normalize_asset(asset_id: str) -> str:
    m = re.search(r"chiller\s*(\d+)", asset_id.lower())
    return f"chiller_{m.group(1)}" if m else asset_id.lower().replace(" ", "_")
