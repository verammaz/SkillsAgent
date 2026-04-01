# skills.py
#
# Seven skills covering the full AssetOpsBench task space.
# Each skill is a self-contained multi-step workflow. Skills:
#   1. Call AssetOpsBench agent tools (tools.py)
#   2. Optionally enrich with Claude reasoning (when ANTHROPIC_API_KEY is set)
#   3. Pull targeted domain knowledge via knowledge.get_knowledge()
#   4. Return a structured dict: {output, confidence, should_stop}
#
# SKILL_REGISTRY (bottom of file) maps skill names to:
#   fn          – the skill function
#   should_skip – callable(context) → bool  (conditional execution)
#   cost        – relative cost 0.0–1.0     (used by executor budget)

import json
import os
import dotenv
dotenv.load_dotenv()  # load .env file if present

from tools import (
    get_sensor_data, get_asset_metadata,
    detect_anomaly, forecast_sensor,
    map_failure, generate_work_order,
)
from knowledge import get_knowledge


# ── LLM helper (gracefully degrades without API key) ─────────────────────────

def _call_claude(system: str, user: str, max_tokens: int = 512) -> str:
    """Call Claude for a reasoning step. Returns raw text or empty string."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return ""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text
    except Exception:
        return ""


def _call_groq(system: str, user: str, max_tokens: int = 512) -> str:
    """Call Claude for a reasoning step. Returns raw text or empty string."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system},
                        {"role": "user", "content": user}])
        return resp.choices[0].message.content
    except Exception:
        return ""


def _parse_json(text: str, fallback: dict) -> dict:
    """Safely parse a JSON block from LLM output."""
    import re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return fallback


# ── Skill 1: Data Retrieval ───────────────────────────────────────────────────

def data_retrieval(asset_id: str, context: dict, task: str) -> dict:
    """IoT agent — fetch sensor time-series data."""
    knowledge = get_knowledge("data_retrieval", task, context)
    lookback  = knowledge.get("time_series_metadata", {}).get("default_lookback_days", 7)
    data      = get_sensor_data(asset_id, lookback_days=lookback)
    return {
        "output":      {"sensor_data": data},
        "confidence":  0.5,   # low: retrieval alone doesn't complete the task
        "should_stop": False,
    }


def _data_retrieval_skip(context: dict) -> bool:
    return False  # always run first for data-dependent queries


# ── Skill 2: Metadata Retrieval ───────────────────────────────────────────────

def metadata_retrieval(asset_id: str, context: dict, task: str) -> dict:
    """IoT agent — fetch sensor catalog and unit descriptions."""
    raw = get_asset_metadata(asset_id)

    # Optionally let Claude summarize for readability
    llm_summary = _call_claude(
        system="Summarize the following sensor catalog in plain language.",
        user=json.dumps(raw),
        max_tokens=256,
    )
    return {
        "output": {
            "metadata": raw,
            "summary":  llm_summary or f"{len(raw.get('sensors', []))} sensors found for {asset_id}.",
        },
        "confidence":  0.95,
        "should_stop": True,   # metadata queries are self-contained
    }


def _metadata_skip(context: dict) -> bool:
    return False


# ── Skill 3: Anomaly Detection ────────────────────────────────────────────────

def anomaly_detection(asset_id: str, context: dict, task: str) -> dict:
    """TSFM agent — detect anomalies in sensor readings."""
    if "sensor_data" not in context:
        return {"output": {}, "confidence": 0.0, "should_stop": False}

    knowledge  = get_knowledge("anomaly_detection", task, context)
    thresholds = knowledge.get("sensor_thresholds")
    result     = detect_anomaly(context["sensor_data"], thresholds)

    severity = result.get("severity", "none")

    # Optionally enrich with Claude reasoning
    llm_out = _call_claude(
        system=(
            "You are an anomaly analysis agent. Given sensor anomalies, "
            "assess their significance. Return JSON with keys: "
            "summary (string), likely_cause (string), urgency (low/medium/high)."
        ),
        user=(
            f"Asset: {asset_id}\n"
            f"Anomalies: {result.get('anomaly_details', [])}\n"
            f"Operating ranges: {knowledge.get('operating_ranges', {})}"
        ),
    )
    enrichment = _parse_json(llm_out, {})

    # No anomalies → task complete (early stop); anomalies → need more skills
    should_stop = severity == "none"
    confidence  = 0.85 if should_stop else 0.6   # 0.6: anomalies found, task not done yet

    return {
        "output": {
            "anomaly_analysis": {**result, **enrichment},
            "anomalies_detected": result["anomalies_detected"],
            "severity":           severity,
        },
        "confidence":  confidence,
        "should_stop": should_stop,
    }


def _anomaly_skip(context: dict) -> bool:
    return "sensor_data" not in context


# ── Skill 4: Root Cause Analysis ──────────────────────────────────────────────

def root_cause_analysis(asset_id: str, context: dict, task: str) -> dict:
    """FMSR agent — map anomaly patterns to failure mode.

    Can run standalone (calls data retrieval + anomaly detection internally)
    or after those skills have already populated context.
    """
    knowledge = get_knowledge("root_cause_analysis", task, context)

    # If executor ran data_retrieval + anomaly_detection first, use their output.
    # Otherwise, run them inline (backwards-compatible with original flat flow).
    if "sensor_data" not in context:
        sensor_data = get_sensor_data(asset_id)
        context["sensor_data"] = sensor_data
    else:
        sensor_data = context["sensor_data"]

    if "anomaly_analysis" not in context:
        thresholds = knowledge.get("sensor_thresholds")
        anomaly    = detect_anomaly(sensor_data, thresholds)
        context["anomaly_analysis"] = anomaly
    else:
        anomaly = context["anomaly_analysis"]

    failure_modes = knowledge.get("failure_modes", [])
    failure       = map_failure(anomaly, failure_modes)
    severity      = anomaly.get("severity", "unknown")

    # llm enrichment: explain the root cause -- use groq (free) for now
    llm_out = _call_groq(
        system=(
            "You are a root cause analysis agent for industrial chillers. "
            "Given anomaly data and a diagnosed failure, explain the root cause "
            "and recommend an action. Return JSON with keys: "
            "explanation (string), recommended_action (string), confidence (0.0-1.0)."
        ),
        user=(
            f"Asset: {asset_id}\n"
            f"Failure: {failure}\n"
            f"Anomalies: {anomaly.get('anomaly_details', [])}\n"
            f"Failure library: {failure_modes}"
        ),
    )
    enrichment = _parse_json(llm_out, {"explanation": "", "recommended_action": "", "confidence": 0.7})
    confidence = float(enrichment.get("confidence", 0.75))

    return {
        "output": {
            "failure":        failure,
            "severity":       severity,
            "anomaly":        anomaly,
            "sensor_data":    sensor_data,
            "rca_detail":     enrichment,
        },
        "confidence":  confidence,
        "should_stop": False,
    }


def _rca_skip(context: dict) -> bool:
    # Skip if anomaly detection already confirmed no anomalies
    return not context.get("anomalies_detected", True) and "anomaly_analysis" in context


# ── Skill 5: Validate Failure ─────────────────────────────────────────────────

def validate_failure(asset_id: str, context: dict, task: str) -> dict:
    """Cross-check failure against maintenance policy; decide if WO is needed."""
    knowledge = get_knowledge("validate_failure", task, context)
    failure   = context.get("failure", "unknown")
    severity  = context.get("severity", "unknown")
    policy    = knowledge.get("maintenance_policy", {})

    # Auto-escalate known critical failure modes
    if failure in policy.get("auto_escalate", []):
        severity = "critical"

    work_order_needed = severity in policy.get("requires_work_order", ["high", "critical"])
    response_time     = policy.get("response_times", {}).get(severity, "TBD")

    return {
        "output": {
            "validated":         True,
            "failure":           failure,
            "severity":          severity,
            "work_order_needed": work_order_needed,
            "response_time":     response_time,
        },
        "confidence":  0.88,
        "should_stop": not work_order_needed,   # stop if no WO needed
    }


def _validate_skip(context: dict) -> bool:
    # Skip if no failure was diagnosed
    return context.get("failure", "unknown") == "unknown" or \
           not context.get("anomalies_detected", True)


# ── Skill 6: Forecasting ──────────────────────────────────────────────────────

def forecasting(asset_id: str, context: dict, task: str) -> dict:
    """TSFM agent — predict future sensor values, flag if maintenance needed."""
    knowledge     = get_knowledge("forecasting", task, context)
    op_ranges     = knowledge.get("operating_ranges", {})
    ts_meta       = knowledge.get("time_series_metadata", {})
    horizon_days  = ts_meta.get("default_lookback_days", 7)

    # Pick the most operationally important sensor for this asset
    sensors  = knowledge.get("sensor_metadata", {}).get("sensors", ["flow_rate_GPM"])
    target   = sensors[0] if sensors else "flow_rate_GPM"

    forecast = forecast_sensor(asset_id, target, horizon_days=horizon_days)

    # Check forecast against operating ranges
    limits   = op_ranges.get(target, {})
    values   = forecast.get("forecasted", [])
    breaches = []
    if limits.get("max") and any(v > limits["max"] for v in values):
        breaches.append(f"{target} forecast exceeds max {limits['max']}")
    if limits.get("min") and any(v < limits["min"] for v in values):
        breaches.append(f"{target} forecast drops below min {limits['min']}")

    maintenance_needed = bool(breaches)

    # Claude enrichment
    llm_out = _call_claude(
        system=(
            "You are a predictive maintenance agent. "
            "Given a sensor forecast and operating limits, explain whether "
            "maintenance is needed. Return JSON with keys: "
            "summary (string), maintenance_needed (bool), urgency (low/medium/high)."
        ),
        user=(
            f"Asset: {asset_id}\nSensor: {target}\n"
            f"Forecast ({horizon_days} days): {values}\n"
            f"Operating range: {limits}\nBreaches: {breaches}"
        ),
    )
    enrichment = _parse_json(llm_out, {"summary": "", "maintenance_needed": maintenance_needed})

    # If forecast is healthy, stop early — no WO needed
    should_stop = not maintenance_needed

    return {
        "output": {
            "forecast":           forecast,
            "breaches":           breaches,
            "maintenance_needed": maintenance_needed,
            "forecast_detail":    enrichment,
        },
        "confidence":  0.85,
        "should_stop": should_stop,
    }


def _forecasting_skip(context: dict) -> bool:
    return False  # forecasting is its own independent flow


# ── Skill 7: Work Order Generation ────────────────────────────────────────────

def work_order_generation(asset_id: str, context: dict, task: str) -> dict:
    """WO agent — create a structured maintenance work order."""
    knowledge = get_knowledge("work_order_generation", task, context)
    failure   = context.get("failure", "unknown")
    severity  = context.get("severity", "high")
    policy    = knowledge.get("maintenance_policy", {})

    # Escalate priority for known critical failures
    priority  = "critical" if failure in policy.get("auto_escalate", []) else severity
    wo        = generate_work_order(asset_id, failure, priority)

    return {
        "output": {
            "work_order": wo,
        },
        "confidence":  0.95,
        "should_stop": True,   # always terminal
    }


def _wo_skip(context: dict) -> bool:
    # Skip if validation said no WO needed
    return not context.get("work_order_needed", True)


# ── Skill registry ────────────────────────────────────────────────────────────
#
# Add new skills here. The executor reads this dict — no other changes needed.

SKILL_REGISTRY = {
    "data_retrieval": {
        "fn":          data_retrieval,
        "should_skip": _data_retrieval_skip,
        "cost":        0.2,
        "description": "Fetch sensor time-series data (IoT agent).",
    },
    "metadata_retrieval": {
        "fn":          metadata_retrieval,
        "should_skip": _metadata_skip,
        "cost":        0.1,
        "description": "Retrieve sensor descriptions and units (IoT agent).",
    },
    "anomaly_detection": {
        "fn":          anomaly_detection,
        "should_skip": _anomaly_skip,
        "cost":        0.7,
        "description": "Detect sensor anomalies using TSFM agent. Requires data_retrieval first.",
    },
    "root_cause_analysis": {
        "fn":          root_cause_analysis,
        "should_skip": _rca_skip,
        "cost":        0.8,
        "description": "Diagnose failure mode using FMSR agent.",
    },
    "validate_failure": {
        "fn":          validate_failure,
        "should_skip": _validate_skip,
        "cost":        0.3,
        "description": "Cross-check failure against maintenance policy.",
    },
    "forecasting": {
        "fn":          forecasting,
        "should_skip": _forecasting_skip,
        "cost":        0.9,
        "description": "Predict future sensor values and flag maintenance need (TSFM agent).",
    },
    "work_order_generation": {
        "fn":          work_order_generation,
        "should_skip": _wo_skip,
        "cost":        0.2,
        "description": "Generate a maintenance work order (WO agent).",
    },
}
