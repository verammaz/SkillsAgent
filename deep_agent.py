import time
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

from tools import (
    get_sensor_data,
    get_asset_metadata,
    detect_anomaly,
    forecast_sensor,
    map_failure,
    generate_work_order,
)


def tool_get_sensor_data(asset_id: str, lookback_days: int = 7) -> dict:
    """Fetch time-series sensor readings for an asset."""
    return get_sensor_data(asset_id, lookback_days=lookback_days)


def tool_get_asset_metadata(asset_id: str) -> dict:
    """Fetch sensor catalog and unit descriptions for an asset."""
    return get_asset_metadata(asset_id)


def tool_detect_anomaly_for_asset(asset_id: str, lookback_days: int = 7) -> dict:
    """Retrieve sensor data for an asset, then run anomaly detection."""
    data = get_sensor_data(asset_id, lookback_days=lookback_days)
    result = detect_anomaly(data)
    return {
        "asset_id": asset_id,
        "lookback_days": lookback_days,
        "anomaly_result": result,
    }


def tool_map_failure_for_asset(asset_id: str, lookback_days: int = 7) -> dict:
    """Retrieve sensor data, detect anomalies, and map them to a likely failure mode."""
    data = get_sensor_data(asset_id, lookback_days=lookback_days)
    anomaly = detect_anomaly(data)
    failure = map_failure(anomaly)
    return {
        "asset_id": asset_id,
        "anomaly_result": anomaly,
        "failure": failure,
    }


def tool_forecast_sensor(asset_id: str, sensor: str, horizon_days: int = 7) -> dict:
    """Forecast future sensor values for an asset."""
    return forecast_sensor(asset_id, sensor, horizon_days=horizon_days)


def tool_generate_work_order(asset_id: str, failure: str, priority: str = "high") -> dict:
    """Generate a work order for a diagnosed failure."""
    return generate_work_order(asset_id, failure, priority)


def _build_agent():
    return create_deep_agent(
        model=init_chat_model("google_genai:gemini-2.5-flash"),
        tools=[
            tool_get_sensor_data,
            tool_get_asset_metadata,
            tool_detect_anomaly_for_asset,
            tool_map_failure_for_asset,
            tool_forecast_sensor,
            tool_generate_work_order,
        ],
        system_prompt=(
            "You are an industrial asset operations assistant. "
            "Use the available tools to inspect asset metadata, fetch sensor data, "
            "detect anomalies, map likely failures, forecast sensor values, "
            "and generate work orders. Prefer tool use over guessing."
        ),
    )


class SkillAgent:
    def __init__(self):
        self.agent = _build_agent()

    def run(self, query: str) -> dict:
        t0 = time.time()

        result = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            }
        )

        latency = round(time.time() - t0, 3)
        final_text = result["messages"][-1].content if result.get("messages") else ""

        return {
            "result": {
                "answer": final_text,
                "raw": result,
            },
            "metrics": {
                "plan": ["deepagents_tool_calling"],
                "skills_executed": [],
                "skipped_conditional": [],
                "skipped_early_stop": [],
                "stopped_at": "final",
                "tool_calls": _count_tool_messages(result),
                "total_cost": "unknown",
                "latency_s": latency,
            },
        }


def _count_tool_messages(result: dict) -> int:
    messages = result.get("messages", [])
    count = 0
    for m in messages:
        if getattr(m, "type", "") == "tool":
            count += 1
        elif m.__class__.__name__ == "ToolMessage":
            count += 1
    return count