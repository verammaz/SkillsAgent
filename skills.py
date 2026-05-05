"""Seven skills covering the AssetOpsBench task space.

Each skill is `fn(asset_id, context, task) -> {output, should_stop}` and is
registered in ``SKILL_REGISTRY`` (bottom of the file) with ``should_skip``,
``cost``, and ``description``. Skills:
    1. Call AssetOpsBench tool wrappers (``tools.py``)
    2. Pull targeted domain knowledge via ``knowledge.get_knowledge()``
    3. Optionally enrich with an LLM call (watsonx → gemini → anthropic → groq)
"""

import json
import os
import dotenv
dotenv.load_dotenv()

from tools import (
    get_sensor_data, get_asset_metadata,
    detect_anomaly, forecast_sensor,
    map_failure_with_meta,
    score_diagnosis_confidence, deep_tsfm_refine_anomalies,
    generate_work_order,
    fetch_tsfm_catalog,
)
from knowledge import get_knowledge
from confidence_evaluator import should_invoke_deep_tsfm, theta_from_env


# ── LLM provider chain ────────────────────────────────────────────────────────
# Preferred order is watsonx → gemini → anthropic → groq; whichever returns a
# non-empty string wins. ``LLM_PROVIDER`` reorders the chain.

_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# ``WATSONX_MODEL_ID`` accepts a comma-separated list; each model is tried in
# order so a transient 500 on one model (e.g. "downstream_request_failed")
# falls through to the next.
_WATSONX_MODELS = [
    m.strip()
    for m in os.getenv(
        "WATSONX_MODEL_ID",
        "meta-llama/llama-4-maverick-17b-128e-instruct-fp8,"
        "meta-llama/llama-3-3-70b-instruct,"
        "mistral-large-2512",
    ).split(",")
    if m.strip()
]
_WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
_WATSONX_MAX_RETRIES = int(os.getenv("WATSONX_MAX_RETRIES", "2"))


def _is_transient_watsonx_error(exc: Exception) -> bool:
    """True for HTTP 5xx / connection errors worth retrying."""
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in (
            "status code: 5",
            "downstream_request_failed",
            "connection refused",
            "timed out",
            "timeout",
            "503",
            "502",
            "500",
        )
    )


def _call_watsonx(system: str, user: str, max_tokens: int = 512) -> str:
    """Call IBM watsonx.ai for a reasoning step. Returns raw text or ""."""
    import time

    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    if not (api_key and project_id):
        return ""
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
    except Exception:
        return ""

    creds = Credentials(url=_WATSONX_URL, api_key=api_key)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_exc: Exception | None = None

    for model_id in _WATSONX_MODELS:
        try:
            model = ModelInference(
                model_id=model_id,
                credentials=creds,
                project_id=project_id,
            )
        except Exception as exc:
            last_exc = exc
            continue

        for attempt in range(_WATSONX_MAX_RETRIES + 1):
            try:
                resp = model.chat(
                    messages=messages,
                    params={"max_tokens": max_tokens, "temperature": 0.0},
                )
                text = (resp["choices"][0]["message"]["content"] or "").strip()
                if text:
                    return text
            except Exception as exc:
                last_exc = exc
                if _is_transient_watsonx_error(exc) and attempt < _WATSONX_MAX_RETRIES:
                    time.sleep(0.6 * (2 ** attempt))
                    continue
                break

        # Final attempt on this model: non-chat generate_text fallback
        try:
            prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
            gen_params = {"max_new_tokens": max_tokens, "decoding_method": "greedy"}
            text = (model.generate_text(prompt=prompt, params=gen_params) or "").strip()
            if text:
                return text
        except Exception as exc:
            last_exc = exc
            continue

    return ""


def _call_gemini(system: str, user: str, max_tokens: int = 512) -> str:
    """Call Gemini for a reasoning step. Returns raw text or empty string."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return ""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
        return (resp.text or "").strip()
    except Exception:
        return ""


def _call_claude(system: str, user: str, max_tokens: int = 512) -> str:
    """Call Claude for a reasoning step. Returns raw text or empty string."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return ""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text
    except Exception:
        return ""


def _call_groq(system: str, user: str, max_tokens: int = 512) -> str:
    """Call Groq (Llama) for a reasoning step. Returns raw text or empty string."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=_GROQ_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


_PROVIDER_FNS = {
    "watsonx": _call_watsonx,
    "gemini": _call_gemini,
    "anthropic": _call_claude,
    "groq": _call_groq,
}

_PROVIDER_ORDER = ("watsonx", "gemini", "anthropic", "groq")


def _call_llm(system: str, user: str, max_tokens: int = 512) -> str:
    """Call the preferred provider, then fall back through the remaining ones."""
    preferred = (os.getenv("LLM_PROVIDER") or "watsonx").lower()
    order = [preferred] + [p for p in _PROVIDER_ORDER if p != preferred]
    for name in order:
        fn = _PROVIDER_FNS.get(name)
        if fn is None:
            continue
        out = fn(system, user, max_tokens)
        if out:
            return out
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
        "should_stop": False,
    }


def _data_retrieval_skip(context: dict) -> bool:
    return False  # always run first for data-dependent queries


# ── Skill 2: Metadata Retrieval ───────────────────────────────────────────────

def metadata_retrieval(asset_id: str, context: dict, task: str) -> dict:
    """IoT + TSFM catalog — sensor catalog plus authoritative AI-task/model lists.

    TSFM lists come from ``servers.tsfm.main`` (parity with MCP ``get_ai_tasks`` /
    ``get_tsfm_models``). Disable via ``TSFM_CATALOG_INJECTION=0``.
    """
    knowledge = get_knowledge("metadata_retrieval", task, context)
    raw = get_asset_metadata(asset_id)

    catalog: dict = {}
    if os.getenv("TSFM_CATALOG_INJECTION", "1").lower() in ("1", "true", "yes"):
        catalog = fetch_tsfm_catalog()

    payload = {
        "user_task": task,
        "asset_sensor_metadata": raw,
        "tsfm_catalog": catalog,
    }
    if knowledge:
        payload["injected_knowledge"] = knowledge

    system = (
        "You are an industrial asset operations assistant. Use the JSON below.\n"
        "- For questions about supported TSFM/AI task types, available pretrained TTM models, "
        "context lengths, or what is (not) supported in the product, follow **tsfm_catalog** only. "
        "Do not invent model_ids; cite task_id / model_id from that list when listing capabilities.\n"
        "- For questions about physical sensors, units, or this asset’s measurement points, use "
        "**asset_sensor_metadata** (and **injected_knowledge** if present).\n"
        "Answer the user_task directly and concisely."
    )
    llm_summary = _call_llm(
        system=system,
        user=json.dumps(payload, default=str),
        max_tokens=512,
    )
    return {
        "output": {
            "metadata": raw,
            "tsfm_catalog": catalog,
            "summary": llm_summary
            or f"{len(raw.get('sensors', []))} sensors for {asset_id}.",
        },
        "should_stop": True,  # metadata / knowledge-style queries are self-contained
    }


def _metadata_skip(context: dict) -> bool:
    return False


# ── Skill 3: Anomaly Detection ────────────────────────────────────────────────

def anomaly_detection(asset_id: str, context: dict, task: str) -> dict:
    """Detect anomalies via profile limits + IQR (ML TSAD only in deep_tsfm_refine_anomalies)."""
    knowledge = get_knowledge("anomaly_detection", task, context)
    if "sensor_data" not in context:
        lookback = knowledge.get("time_series_metadata", {}).get(
            "default_lookback_days", 7
        )
        context["sensor_data"] = get_sensor_data(asset_id, lookback_days=lookback)

    thresholds = knowledge.get("sensor_thresholds")
    result = detect_anomaly(context["sensor_data"], thresholds)

    severity = result.get("severity", "none")

    # Optionally enrich with LLM reasoning
    llm_out = _call_llm(
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

    return {
        "output": {
            "anomaly_analysis": {**result, **enrichment},
            "anomalies_detected": result["anomalies_detected"],
            "severity":           severity,
        },
        "should_stop": should_stop,
    }


def _anomaly_skip(context: dict) -> bool:
    return False


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
    sensor_data   = context["sensor_data"]

    meta = map_failure_with_meta(anomaly, failure_modes, asset_id=asset_id)
    failure = meta["failure"]
    wo_history = knowledge.get("work_order_history")
    confidence_pre = score_diagnosis_confidence(
        anomaly, meta, sensor_data=sensor_data, wo_history=wo_history, task=task
    )
    confidence = confidence_pre

    anomaly_def = knowledge.get("anomaly_definition")
    theta = theta_from_env()
    deep_invoked = False
    if should_invoke_deep_tsfm(confidence_pre, theta=theta):
        anomaly = deep_tsfm_refine_anomalies(
            asset_id,
            sensor_data,
            anomaly,
            anomaly_definition=anomaly_def if isinstance(anomaly_def, dict) else None,
        )
        meta = map_failure_with_meta(anomaly, failure_modes, asset_id=asset_id)
        failure = meta["failure"]
        confidence = score_diagnosis_confidence(
            anomaly, meta, sensor_data=sensor_data, wo_history=wo_history, task=task
        )
        deep_invoked = True

    severity = anomaly.get("severity", "unknown")

    llm_out = _call_llm(
        system=(
            "You are a root cause analysis agent for industrial chillers. "
            "Given anomaly data and a diagnosed failure, explain the root cause "
            "and recommend an action. Return JSON with keys: "
            "explanation (string), recommended_action (string)."
        ),
        user=(
            f"Asset: {asset_id}\n"
            f"Failure: {failure}\n"
            f"Diagnosis confidence: {confidence:.2f}\n"
            f"Deep TSFM refinement run: {deep_invoked}\n"
            f"Anomalies: {anomaly.get('anomaly_details', [])}\n"
            f"Failure library: {failure_modes}"
        ),
    )
    enrichment = _parse_json(llm_out, {"explanation": "", "recommended_action": ""})

    return {
        "output": {
            "failure":                       failure,
            "severity":                      severity,
            "diagnosis_confidence":          round(confidence, 3),
            "diagnosis_confidence_pre_deep": round(confidence_pre, 3),
            "deep_tsfm_invoked":             deep_invoked,
            "anomaly_analysis":              anomaly,
            "anomalies_detected":            anomaly.get("anomalies_detected", False),
            "rca_detail":                    enrichment,
        },
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
        "should_stop": not work_order_needed,   # stop if no WO needed
    }


def _validate_skip(context: dict) -> bool:
    # Skip if no failure was diagnosed, or anomaly detection confirmed none
    if context.get("failure", "unknown") == "unknown":
        return True
    if "anomalies_detected" in context and not context["anomalies_detected"]:
        return True
    return False


# ── Skill 6: Forecasting ──────────────────────────────────────────────────────

def forecasting(asset_id: str, context: dict, task: str) -> dict:
    """TSFM agent — predict future sensor values, flag if maintenance needed."""
    knowledge = get_knowledge("forecasting", task, context)
    op_ranges = knowledge.get("operating_ranges", {})
    ts_meta = knowledge.get("time_series_metadata", {})
    horizon_days = ts_meta.get("default_lookback_days", 7)
    lookback = ts_meta.get("default_lookback_days", 7)

    if "sensor_data" not in context:
        context["sensor_data"] = get_sensor_data(asset_id, lookback_days=lookback)
    sensor_data = context["sensor_data"]

    # Pick the most operationally important sensor for this asset
    sensors = knowledge.get("sensor_metadata", {}).get("sensors", ["flow_rate_GPM"])
    target = sensors[0] if sensors else "flow_rate_GPM"

    forecast = forecast_sensor(
        asset_id,
        target,
        horizon_days=horizon_days,
        sensor_data=sensor_data,
        task=task,
    )

    # Check forecast against operating ranges
    limits   = op_ranges.get(target, {})
    values   = forecast.get("forecasted", [])
    breaches = []
    if limits.get("max") and any(v > limits["max"] for v in values):
        breaches.append(f"{target} forecast exceeds max {limits['max']}")
    if limits.get("min") and any(v < limits["min"] for v in values):
        breaches.append(f"{target} forecast drops below min {limits['min']}")

    maintenance_needed = bool(breaches)

    # LLM enrichment
    llm_out = _call_llm(
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
        "description": "Sensor catalog + TSFM task/model list (AssetOps TSFM server parity).",
    },
    "anomaly_detection": {
        "fn":          anomaly_detection,
        "should_skip": _anomaly_skip,
        "cost":        0.7,
        "description": "Detect sensor anomalies (profile + IQR). Fetches IoT data if missing.",
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
        "description": "Predict future sensor values (TSFM subprocess when IoT context is long enough).",
    },
    "work_order_generation": {
        "fn":          work_order_generation,
        "should_skip": _wo_skip,
        "cost":        0.2,
        "description": "Generate a maintenance work order (WO agent).",
    },
}


# ── Calibrated-cost loader ────────────────────────────────────────────────────
#
# If ``SKILL_COSTS_PATH`` (or ``./skill_costs.json``) exists, we override the
# hand-assigned ``cost`` priors in :data:`SKILL_REGISTRY` with measured median
# wall-clock seconds from ``scripts/calibrate_costs.py``. The same file may
# carry ``__deep_tsfm__`` which is consumed by ``agent.DEEP_TSFM_COST``.
# Missing or unparseable files are ignored silently — priors win.

def _load_calibrated_costs() -> dict:
    import json as _json
    import os as _os
    from pathlib import Path as _Path

    path = _os.getenv("SKILL_COSTS_PATH") or "skill_costs.json"
    p = _Path(path)
    if not p.is_file():
        return {}
    try:
        data = _json.loads(p.read_text())
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    for name, cost in data.items():
        if name.startswith("__"):
            continue
        if name in SKILL_REGISTRY and isinstance(cost, (int, float)):
            SKILL_REGISTRY[name]["cost"] = float(cost)
    return data


CALIBRATED_COSTS = _load_calibrated_costs()
