"""deep_agent.py — DeepAgents-based orchestrator for AssetOpsBench.

Uses ``create_deep_agent`` from the ``deepagents`` package to orchestrate the
real AssetOpsBench tool wrappers defined in ``tools.py`` / ``skills.py``.

Key design decisions
---------------------
* All tool bodies reuse the real implementations from ``skills.py`` and
  ``tools.py``; no business logic is duplicated here.
* The confidence-gated deep-TSFM decision is implemented **inside**
  ``_fmsr_root_cause_body`` in Python (not delegated to the LLM) so the
  threshold gate is deterministic and auditable.
* The LangChain tool trace is parsed after invocation to populate a
  ``metrics`` dict that is compatible with the project's evaluator /
  ``run.py`` format.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import dotenv
dotenv.load_dotenv()

from langchain_core.tools import tool

from tools import (
    get_sensor_data,
    get_asset_metadata,
    detect_anomaly,
    forecast_sensor,
    map_failure_with_meta,
    score_diagnosis_confidence,
    deep_tsfm_refine_anomalies,
    generate_work_order,
)
from knowledge import get_knowledge
from confidence_evaluator import should_invoke_deep_tsfm, theta_from_env

from deepagents import create_deep_agent

logger = logging.getLogger(__name__)

# ── LLM model selection ───────────────────────────────────────────────────────
# Uses the same provider-fallback order as skills.py (watsonx → gemini → …).
# We keep all imports lazy so the file imports without crashing when no
# provider package is installed.

_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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


def _init_model():
    """Return a LangChain BaseChatModel, trying providers in priority order.

    Mirrors the fallback chain in skills.py: watsonx → gemini → anthropic → groq.
    ``LLM_PROVIDER`` env var reorders the chain exactly as in skills.py.
    """
    preferred = (os.getenv("LLM_PROVIDER") or "watsonx").lower()

    def _try_watsonx():
        api_key = os.getenv("WATSONX_API_KEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        if not (api_key and project_id):
            return None
        try:
            from langchain_ibm import ChatWatsonx
            # Try each model in the comma-separated list in order.
            for model_id in _WATSONX_MODELS:
                try:
                    return ChatWatsonx(
                        model_id=model_id,
                        url=_WATSONX_URL,
                        api_key=api_key,
                        project_id=project_id,
                    )
                except Exception as exc:
                    logger.debug("WatsonX model %s init failed: %s", model_id, exc)
        except ImportError:
            logger.debug("langchain_ibm not installed; skipping watsonx")
        return None

    def _try_gemini():
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=_GEMINI_MODEL, google_api_key=api_key)
        except Exception as exc:
            logger.debug("Gemini init failed: %s", exc)
        return None

    def _try_anthropic():
        if not os.getenv("ANTHROPIC_API_KEY"):
            return None
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
            )
        except Exception as exc:
            logger.debug("Anthropic init failed: %s", exc)
        return None

    def _try_groq():
        if not os.getenv("GROQ_API_KEY"):
            return None
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
        except Exception as exc:
            logger.debug("Groq init failed: %s", exc)
        return None

    _PROVIDERS = {
        "watsonx":   _try_watsonx,
        "gemini":    _try_gemini,
        "anthropic": _try_anthropic,
        "groq":      _try_groq,
    }
    _ORDER = ("watsonx", "gemini", "anthropic", "groq")

    order = [preferred] + [p for p in _ORDER if p != preferred]
    for name in order:
        fn = _PROVIDERS.get(name)
        if fn is None:
            continue
        model = fn()
        if model is not None:
            logger.debug("LLM provider selected: %s", name)
            return model

    # Last resort: langchain generic init_chat_model
    try:
        from langchain.chat_models import init_chat_model
        return init_chat_model(f"google_genai:{_GEMINI_MODEL}")
    except Exception as exc:
        raise RuntimeError(
            "No LLM provider could be initialised. "
            "Set WATSONX_API_KEY+WATSONX_PROJECT_ID, GEMINI_API_KEY, "
            "ANTHROPIC_API_KEY, or GROQ_API_KEY."
        ) from exc


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool
def iot_data_retrieval_tool(asset_id: str, lookback_days: int = 7) -> str:
    """Fetch time-series sensor readings for an asset from the IoT agent.

    Args:
        asset_id: Asset identifier, e.g. 'Chiller 6' or 'chiller_9'.
        lookback_days: History window in days (default 7).

    Returns:
        JSON string with keys: asset_id, lookback_days, readings, source.
    """
    try:
        data = get_sensor_data(asset_id, lookback_days=lookback_days)
        return json.dumps(data, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "tool": "iot_data_retrieval_tool"})


@tool
def sensor_metadata_tool(asset_id: str) -> str:
    """Fetch the sensor catalog and unit descriptions for an asset.

    Args:
        asset_id: Asset identifier, e.g. 'Chiller 6'.

    Returns:
        JSON string with keys: asset_id, sensors (list of name/unit/description).
    """
    try:
        meta = get_asset_metadata(asset_id)
        # Optionally enrich via skills.py LLM helper
        try:
            from skills import _call_llm
            summary = _call_llm(
                "Summarize the following sensor catalog in plain language.",
                json.dumps(meta),
                max_tokens=256,
            )
            if summary:
                meta["summary"] = summary
        except Exception:
            pass
        return json.dumps(meta, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "tool": "sensor_metadata_tool"})


@tool
def lightweight_anomaly_tool(asset_id: str, lookback_days: int = 7) -> str:
    """Retrieve sensor data then run lightweight anomaly detection (profile + IQR).

    This is the fast, cheap first-pass anomaly check that always runs for
    fault-diagnosis and anomaly-detection queries.

    Args:
        asset_id: Asset identifier.
        lookback_days: Sensor history window in days.

    Returns:
        JSON string with keys: asset_id, anomalies_detected, anomaly_details,
        severity, sensor_data (embedded for downstream tools).
    """
    try:
        knowledge = get_knowledge("anomaly_detection", asset_id, {})
        lookback = knowledge.get("time_series_metadata", {}).get(
            "default_lookback_days", lookback_days
        )
        data = get_sensor_data(asset_id, lookback_days=lookback)
        thresholds = knowledge.get("sensor_thresholds")
        result = detect_anomaly(data, thresholds)
        result["asset_id"] = asset_id
        result["sensor_data"] = data   # embed for fmsr_root_cause_tool
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "tool": "lightweight_anomaly_tool"})


@tool
def fmsr_root_cause_tool(
    asset_id: str,
    anomaly_json: str,
    sensor_data_json: str,
    task: str = "",
    theta: float = -1.0,
) -> str:
    """Run FMSR root-cause analysis on detected anomalies.

    Implements the full conditional deep-TSFM gate deterministically in Python:
      1. Map anomaly → failure mode (FMSR agent).
      2. Score diagnosis confidence.
      3. If confidence < theta → run deep_tsfm_refine_anomalies → re-map.
      4. Return failure, confidence, deep_tsfm_invoked flag.

    Args:
        asset_id: Asset identifier.
        anomaly_json: JSON string output from lightweight_anomaly_tool.
        sensor_data_json: JSON string of sensor data (readings dict).
        task: Original user query (used for task-specificity scoring).
        theta: Confidence threshold (default reads RCA_CONFIDENCE_THETA env var).
    """
    try:
        anomaly = json.loads(anomaly_json) if isinstance(anomaly_json, str) else anomaly_json
        sensor_data = (
            json.loads(sensor_data_json) if isinstance(sensor_data_json, str) else sensor_data_json
        )

        # Pull from embedded sensor_data if separate arg was not passed
        if not sensor_data and anomaly.get("sensor_data"):
            sensor_data = anomaly.pop("sensor_data")

        knowledge = get_knowledge("root_cause_analysis", task, {})
        failure_modes = knowledge.get("failure_modes", [])
        wo_history = knowledge.get("work_order_history")
        anomaly_def = knowledge.get("anomaly_definition")

        # 1. First-pass FMSR mapping
        meta = map_failure_with_meta(anomaly, failure_modes, asset_id=asset_id)
        failure = meta["failure"]

        # 2. Confidence scoring
        confidence_pre = score_diagnosis_confidence(
            anomaly, meta, sensor_data=sensor_data, wo_history=wo_history, task=task
        )
        confidence = confidence_pre

        # 3. Deterministic deep-TSFM gate
        eff_theta = theta if theta > 0 else theta_from_env()
        deep_invoked = False
        if should_invoke_deep_tsfm(confidence_pre, theta=eff_theta):
            anomaly = deep_tsfm_refine_anomalies(
                asset_id,
                sensor_data,
                anomaly,
                anomaly_definition=anomaly_def if isinstance(anomaly_def, dict) else None,
            )
            # 4. Re-run FMSR on refined anomalies
            meta = map_failure_with_meta(anomaly, failure_modes, asset_id=asset_id)
            failure = meta["failure"]
            confidence = score_diagnosis_confidence(
                anomaly, meta, sensor_data=sensor_data, wo_history=wo_history, task=task
            )
            deep_invoked = True

        # Optional LLM enrichment (best-effort, non-blocking)
        rca_detail = {}
        try:
            from skills import _call_llm, _parse_json
            llm_out = _call_llm(
                "You are a root cause analysis agent for industrial chillers. "
                "Given anomaly data and a diagnosed failure, explain the root cause "
                "and recommend an action. Return JSON with keys: "
                "explanation (string), recommended_action (string).",
                f"Asset: {asset_id}\nFailure: {failure}\n"
                f"Confidence: {confidence:.2f}\nDeep TSFM run: {deep_invoked}\n"
                f"Anomalies: {anomaly.get('anomaly_details', [])}",
            )
            rca_detail = _parse_json(llm_out, {})
        except Exception:
            pass

        return json.dumps({
            "asset_id": asset_id,
            "failure": failure,
            "severity": anomaly.get("severity", "unknown"),
            "diagnosis_confidence": round(confidence, 3),
            "diagnosis_confidence_pre_deep": round(confidence_pre, 3),
            "deep_tsfm_invoked": deep_invoked,
            "anomalies_detected": anomaly.get("anomalies_detected", False),
            "anomaly_analysis": anomaly,
            "rca_detail": rca_detail,
            "matched_via": meta.get("matched_via", "unknown"),
        }, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "tool": "fmsr_root_cause_tool"})


@tool
def forecasting_tool(asset_id: str, sensor: str = "flow_rate_GPM", horizon_days: int = 7) -> str:
    """Forecast future sensor values using the TSFM agent.

    Checks forecast against operating ranges and flags maintenance need.

    Args:
        asset_id: Asset identifier.
        sensor: Sensor to forecast (default 'flow_rate_GPM').
        horizon_days: Forecast horizon in days (default 7).

    Returns:
        JSON string with keys: forecast, breaches, maintenance_needed, forecast_detail.
    """
    try:
        knowledge = get_knowledge("forecasting", asset_id, {})
        op_ranges = knowledge.get("operating_ranges", {})
        ts_meta = knowledge.get("time_series_metadata", {})
        lookback = ts_meta.get("default_lookback_days", 7)

        # Use knowledge-suggested sensor if caller left default and knowledge has one
        sensors_hint = knowledge.get("sensor_metadata", {}).get("sensors", [])
        if sensor == "flow_rate_GPM" and sensors_hint:
            sensor = sensors_hint[0]

        data = get_sensor_data(asset_id, lookback_days=lookback)
        fc = forecast_sensor(asset_id, sensor, horizon_days=horizon_days, sensor_data=data)

        limits = op_ranges.get(sensor, {})
        values = fc.get("forecasted", [])
        breaches: list[str] = []
        if limits.get("max") and any(v > limits["max"] for v in values):
            breaches.append(f"{sensor} forecast exceeds max {limits['max']}")
        if limits.get("min") and any(v < limits["min"] for v in values):
            breaches.append(f"{sensor} forecast drops below min {limits['min']}")

        maintenance_needed = bool(breaches)

        # Optional LLM enrichment
        enrichment: dict = {}
        try:
            from skills import _call_llm, _parse_json
            llm_out = _call_llm(
                "You are a predictive maintenance agent. Given a sensor forecast and "
                "operating limits, explain whether maintenance is needed. Return JSON "
                "with keys: summary (string), maintenance_needed (bool), urgency (low/medium/high).",
                f"Asset: {asset_id}\nSensor: {sensor}\n"
                f"Forecast ({horizon_days}d): {values}\n"
                f"Operating range: {limits}\nBreaches: {breaches}",
            )
            enrichment = _parse_json(llm_out, {})
        except Exception:
            pass

        return json.dumps({
            "asset_id": asset_id,
            "sensor": sensor,
            "forecast": fc,
            "breaches": breaches,
            "maintenance_needed": maintenance_needed,
            "forecast_detail": enrichment,
        }, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "tool": "forecasting_tool"})


@tool
def work_order_tool(asset_id: str, failure: str, priority: str = "high") -> str:
    """Generate a structured maintenance work order via the WO agent.

    Args:
        asset_id: Asset identifier.
        failure: Diagnosed failure mode string.
        priority: Work order priority (high, medium, low, critical).

    Returns:
        JSON string representing the work order.
    """
    try:
        knowledge = get_knowledge("work_order_generation", asset_id, {})
        policy = knowledge.get("maintenance_policy", {})
        eff_priority = "critical" if failure in policy.get("auto_escalate", []) else priority
        wo = generate_work_order(asset_id, failure, eff_priority)
        return json.dumps(wo, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "tool": "work_order_tool"})


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an AssetOpsBench industrial operations agent. "
    "Use the available tools to answer maintenance, diagnosis, anomaly, "
    "forecasting, metadata, and work-order queries. "
    "Prefer low-cost tools first. "
    "For fault diagnosis, always retrieve relevant data, run lightweight "
    "anomaly detection, then run FMSR root-cause analysis. "
    "Only invoke full/deep TSFM validation when FMSR confidence is below "
    "the configured threshold (this is handled automatically inside "
    "fmsr_root_cause_tool — you do not need to call a separate deep-TSFM tool). "
    "If deep TSFM is invoked, re-run FMSR using the refined anomaly evidence "
    "before generating a work order. "
    "For simple metadata queries, only call sensor_metadata_tool. "
    "For anomaly-only queries, stop after lightweight_anomaly_tool if no "
    "anomalies are found. "
    "For forecasting queries, generate a work order only if forecasted values "
    "violate operating ranges or maintenance policy. "
    "Always return a final answer plus trace metadata: tools_called, "
    "skills_skipped, confidence, tsfm_deep_invoked, latency, and final "
    "recommendation."
)

_ALL_TOOLS = [
    iot_data_retrieval_tool,
    sensor_metadata_tool,
    lightweight_anomaly_tool,
    fmsr_root_cause_tool,
    forecasting_tool,
    work_order_tool,
]

# Ordered list of all tool names (used for skipped-tool inference)
_ALL_TOOL_NAMES = [t.name for t in _ALL_TOOLS]


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_deep_agent(theta: float | None = None):
    """Build and return a compiled DeepAgent.

    Args:
        theta: Optional confidence threshold override (default reads env).

    Returns:
        A compiled LangGraph agent ready to ``.invoke()``.
    """
    model = _init_model()
    agent = create_deep_agent(
        model=model,
        tools=_ALL_TOOLS,
        system_prompt=_SYSTEM_PROMPT,
    )
    return agent


# ── Output parsing helpers ────────────────────────────────────────────────────

def _parse_tool_calls(messages: list) -> tuple[list[str], dict[str, Any]]:
    """Return (ordered tool names called, merged payload from tool outputs)."""
    tools_called: list[str] = []
    payload: dict[str, Any] = {}

    # keyed accumulator: tool_name → last output dict from that tool
    per_tool: dict[str, dict] = {}
    call_id_to_name: dict[str, str] = {}

    for msg in messages:
        # AIMessage with tool_calls attribute — record names and call IDs
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            for tc in tcs:
                if isinstance(tc, dict):
                    name, call_id = tc.get("name"), tc.get("id", "")
                else:
                    name, call_id = getattr(tc, "name", None), getattr(tc, "id", "") or ""
                if name:
                    tools_called.append(name)
                    if call_id:
                        call_id_to_name[call_id] = name

        # ToolMessage — parse and store under the producing tool's name
        msg_type = getattr(msg, "type", "") or msg.__class__.__name__
        if msg_type in ("tool", "ToolMessage"):
            content = getattr(msg, "content", "") or ""
            call_id = getattr(msg, "tool_call_id", "") or ""
            tool_name = call_id_to_name.get(call_id) or (tools_called[-1] if tools_called else "unknown")
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    per_tool[tool_name] = parsed
                    payload.update(parsed)
            except (json.JSONDecodeError, TypeError):
                pass

    # Prefer fmsr_root_cause_tool output for confidence/deep_tsfm fields
    # so later tool messages (e.g. work_order_tool) do not overwrite them.
    for key in ("diagnosis_confidence", "diagnosis_confidence_pre_deep",
                "deep_tsfm_invoked", "failure", "anomaly_analysis",
                "anomalies_detected", "severity"):
        if key in per_tool.get("fmsr_root_cause_tool", {}):
            payload[key] = per_tool["fmsr_root_cause_tool"][key]

    return tools_called, payload



def _extract_metrics(
    messages: list,
    t0: float,
) -> tuple[dict, dict]:
    """Return (result dict, metrics dict) compatible with evaluator/run.py."""
    tools_called, payload = _parse_tool_calls(messages)

    # Final answer text
    final_text = ""
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", "") or msg.__class__.__name__
        if msg_type in ("ai", "AIMessage"):
            ct = getattr(msg, "content", "")
            if ct and isinstance(ct, str):
                final_text = ct
                break

    # Infer skipped tools
    skills_skipped = [n for n in _ALL_TOOL_NAMES if n not in tools_called]

    deep_tsfm = bool(payload.get("deep_tsfm_invoked", False))
    confidence = payload.get("diagnosis_confidence")
    confidence_pre = payload.get("diagnosis_confidence_pre_deep", confidence)

    # Map tool names to skill registry names for evaluator compatibility
    _TOOL_TO_SKILL = {
        "iot_data_retrieval_tool": "data_retrieval",
        "sensor_metadata_tool": "metadata_retrieval",
        "lightweight_anomaly_tool": "anomaly_detection",
        "fmsr_root_cause_tool": "root_cause_analysis",
        "forecasting_tool": "forecasting",
        "work_order_tool": "work_order_generation",
    }
    skill_names_called = [_TOOL_TO_SKILL.get(n, n) for n in tools_called]
    skill_names_skipped = [_TOOL_TO_SKILL.get(n, n) for n in skills_skipped]

    latency = round(time.time() - t0, 3)

    result = {
        "answer": final_text,
        # Surface key payload fields at the top level for evaluator artefact checks
        "failure": payload.get("failure"),
        "anomaly_analysis": payload.get("anomaly_analysis") or payload.get("anomalies_detected"),
        "work_order": payload.get("work_order_id"),
        "forecast": payload.get("forecast"),
        "metadata": payload.get("sensors"),
        "raw": payload,
    }

    metrics = {
        "plan": skill_names_called,
        "skills_executed": skill_names_called,
        "skipped_conditional": [],       # deep_agent does not use should_skip predicates
        "skipped_early_stop": skill_names_skipped,
        "skills_skipped": skill_names_skipped,
        "stopped_at": skill_names_called[-1] if skill_names_called else "none",
        "tool_calls": len(tools_called),
        "total_cost": round(
            sum(
                {"iot_data_retrieval_tool": 0.2, "sensor_metadata_tool": 0.1,
                 "lightweight_anomaly_tool": 0.7, "fmsr_root_cause_tool": 0.8,
                 "forecasting_tool": 0.9, "work_order_tool": 0.2}.get(n, 0.3)
                for n in tools_called
            ),
            3,
        ),
        "latency_s": latency,
        "deep_tsfm_invoked": deep_tsfm,
        "diagnosis_confidence": confidence,
        "diagnosis_confidence_pre_deep": confidence_pre,
        "tsfm_deep_invoked": deep_tsfm,      # alias used by proposal
        "confidence": confidence,             # alias
    }

    return result, metrics


# ── SkillAgent-compatible wrapper ─────────────────────────────────────────────

class SkillAgent:
    """Drop-in replacement for the skills-based SkillAgent using DeepAgents.

    Exposes the same ``.run(query) -> {result, metrics}`` interface so
    ``run.py``, ``eval_runner.py``, and any other callers work unchanged.
    """

    def __init__(self, cost_budget: float | None = None, theta: float | None = None):
        self._theta = theta
        self._cost_budget = cost_budget  # stored for metadata; not yet enforced
        self._agent = build_deep_agent(theta=theta)

    def run(self, query: str) -> dict:
        t0 = time.time()

        try:
            state = self._agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
        except Exception as exc:
            latency = round(time.time() - t0, 3)
            return {
                "result": {"answer": f"Agent error: {exc}", "error": str(exc)},
                "metrics": {
                    "plan": [],
                    "skills_executed": [],
                    "skipped_conditional": [],
                    "skipped_early_stop": [],
                    "skills_skipped": _ALL_TOOL_NAMES,
                    "stopped_at": "error",
                    "tool_calls": 0,
                    "total_cost": 0.0,
                    "latency_s": latency,
                    "deep_tsfm_invoked": False,
                    "diagnosis_confidence": None,
                    "diagnosis_confidence_pre_deep": None,
                    "tsfm_deep_invoked": False,
                    "confidence": None,
                },
            }

        messages = state.get("messages", [])
        result, metrics = _extract_metrics(messages, t0)
        return {"result": result, "metrics": metrics}


# ── Convenience entry point ───────────────────────────────────────────────────

def run_deep_agent(query: str, threshold: float | None = None) -> dict:
    """Run a single query through the DeepAgent and return {result, metrics}.

    Args:
        query: Natural-language maintenance / operations query.
        threshold: Optional confidence threshold override (default from env).

    Returns:
        Dict with keys ``result`` and ``metrics`` matching the evaluator format.
    """
    return SkillAgent(theta=threshold).run(query)