# knowledge.py
#
# Knowledge plugins: each plugin encapsulates one domain knowledge source and
# injects ONLY what a given skill actually needs, keeping context windows lean.
#
# Interface (every plugin must implement):
#   .name             str       unique identifier
#   .relevant_skills  set[str]  skills this plugin serves (empty = all)
#   .retrieve(skill_name, task, context) -> dict
#
# Add new plugins to ALL_PLUGINS — the executor picks them up automatically.

import re


# ── Plugin 1: Sensor Metadata ─────────────────────────────────────────────────

class SensorMetadataPlugin:
    """Sensor catalog: names, units, physical descriptions per asset."""

    name = "sensor_metadata"
    relevant_skills = {"data_retrieval", "anomaly_detection", "metadata_retrieval"}

    CATALOG = {
        "chiller_6": {
            "sensors": ["vibration_mm_s", "CHW_supply_temp_F", "CHW_return_temp_F",
                        "compressor_power_kW", "flow_rate_GPM"],
            "units": {"vibration": "mm/s", "temp": "°F", "power": "kW", "flow": "GPM"},
        },
        "chiller_9": {
            "sensors": ["condenser_flow_GPM", "compressor_speed_rpm",
                        "refrigerant_psi", "CHW_supply_temp_F", "COP"],
            "units": {"flow": "GPM", "speed": "RPM", "pressure": "PSI", "temp": "°F"},
        },
    }

    def retrieve(self, skill_name: str, task: str, context: dict) -> dict:
        if not _relevant(skill_name, self.relevant_skills):
            return {}
        asset = _extract_asset(task)
        return {
            "asset_id":       asset,
            "sensor_metadata": self.CATALOG.get(asset, {}),
        }


# ── Plugin 2: Failure Mode Library ───────────────────────────────────────────

class FailureModePlugin:
    """Known failure modes with associated symptoms — used by FMSR agent."""

    name = "failure_modes"
    relevant_skills = {"root_cause_analysis", "anomaly_detection", "validate_failure"}

    FAILURE_MODES = [
        {"name": "cavitation",        "symptoms": ["vibration", "flow"],           "severity": "high"},
        {"name": "bearing_failure",   "symptoms": ["vibration", "power", "temp"],  "severity": "high"},
        {"name": "condenser_fouling", "symptoms": ["temp", "power", "efficiency"], "severity": "medium"},
        {"name": "refrigerant_leak",  "symptoms": ["pressure", "temp", "COP"],     "severity": "high"},
        {"name": "sensor_drift",      "symptoms": ["offset", "flat", "inconsistent"], "severity": "low"},
    ]

    def retrieve(self, skill_name: str, task: str, context: dict) -> dict:
        if not _relevant(skill_name, self.relevant_skills):
            return {}
        return {"failure_modes": self.FAILURE_MODES}


# ── Plugin 3: Maintenance Policy ─────────────────────────────────────────────

class MaintenancePolicyPlugin:
    """When a work order is required, response times, escalation rules."""

    name = "maintenance_policy"
    relevant_skills = {"validate_failure", "work_order_generation"}

    POLICY = {
        "requires_work_order":  ["high", "critical"],
        "monitor_only":         ["low", "medium"],
        "response_times": {
            "critical": "4 hours",
            "high":     "24 hours",
            "medium":   "72 hours",
            "low":      "next scheduled maintenance",
        },
        "auto_escalate": ["cavitation", "bearing_failure", "refrigerant_leak"],
    }

    def retrieve(self, skill_name: str, task: str, context: dict) -> dict:
        if not _relevant(skill_name, self.relevant_skills):
            return {}
        return {"maintenance_policy": self.POLICY}


# ── Plugin 4: Operating Ranges ───────────────────────────────────────────────

class OperatingRangesPlugin:
    """Normal sensor operating ranges — used as anomaly detection thresholds."""

    name = "operating_ranges"
    relevant_skills = {"anomaly_detection", "forecasting", "root_cause_analysis"}

    RANGES = {
        "chiller_6": {
            "vibration_mm_s":      {"max": 4.5},
            "CHW_supply_temp_F":   {"min": 42.0, "max": 46.0},
            "CHW_return_temp_F":   {"min": 52.0, "max": 56.0},
            "compressor_power_kW": {"max": 250.0},
            "flow_rate_GPM":       {"min": 850.0},
        },
        "chiller_9": {
            "condenser_flow_GPM":   {"min": 700.0},
            "compressor_speed_rpm": {"max": 3500.0},
            "refrigerant_psi":      {"min": 60.0, "max": 120.0},
        },
    }

    def retrieve(self, skill_name: str, task: str, context: dict) -> dict:
        if not _relevant(skill_name, self.relevant_skills):
            return {}
        asset = _extract_asset(task)
        ranges = self.RANGES.get(asset, self.RANGES["chiller_6"])
        return {
            "operating_ranges":  ranges,
            "sensor_thresholds": ranges,  # alias used by detect_anomaly()
        }


# ── Plugin 5: Time-Series Metadata ───────────────────────────────────────────

class TimeSeriesMetadataPlugin:
    """Seasonality, resolution, and feature config for the TSFM agent."""

    name = "time_series_metadata"
    relevant_skills = {"forecasting", "data_retrieval"}

    TS_META = {
        "default_resolution":   "5 minutes",
        "default_lookback_days": 30,
        "seasonality":          {"daily": True, "weekly": True},
        "lag_features":         [1, 6, 12, 24, 48],
        "rolling_windows":      [12, 24, 72],
    }

    def retrieve(self, skill_name: str, task: str, context: dict) -> dict:
        if not _relevant(skill_name, self.relevant_skills):
            return {}
        return {"time_series_metadata": self.TS_META}


# ── Plugin registry & merge function ─────────────────────────────────────────

ALL_PLUGINS = [
    SensorMetadataPlugin(),
    FailureModePlugin(),
    MaintenancePolicyPlugin(),
    OperatingRangesPlugin(),
    TimeSeriesMetadataPlugin(),
]


def get_knowledge(skill_name: str, task: str, context: dict) -> dict:
    """Merge knowledge from every plugin that serves `skill_name`."""
    merged = {}
    for plugin in ALL_PLUGINS:
        if _relevant(skill_name, plugin.relevant_skills):
            try:
                merged.update(plugin.retrieve(skill_name, task, context))
            except Exception as e:
                pass  # degrade gracefully — missing knowledge ≠ failure
    return merged


# ── Shared helpers ────────────────────────────────────────────────────────────

def _relevant(skill_name: str, relevant_skills: set) -> bool:
    return not relevant_skills or skill_name in relevant_skills


def _extract_asset(task: str) -> str:
    m = re.search(r"chiller\s*(\d+)", task.lower())
    return f"chiller_{m.group(1)}" if m else "chiller_6"
