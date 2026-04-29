"""Smoke tests for deep_agent.py — verifies the real tools are wired correctly.

Run with:
    python -m pytest tests/test_deep_agent_smoke.py -v
or:
    python tests/test_deep_agent_smoke.py
"""

import json
import os
import sys
import types

# ── minimal stubs so the file imports without a live backend ──────────────────

# Stub deepagents with a fake agent that immediately returns a canned answer
try:
    from deepagents import create_deep_agent  # noqa: F401
    _HAVE_DEEPAGENTS = True
except ImportError:
    _HAVE_DEEPAGENTS = False


class _FakeAIMessage:
    type = "ai"
    tool_calls = []
    def __init__(self, content=""):
        self.content = content


def _make_stub_deepagents():
    deepagents_mod = types.ModuleType("deepagents")
    def _stub_create(*a, **kw):
        class _FakeAgent:
            def invoke(self, state):
                return {"messages": [_FakeAIMessage("Stub agent answer.")]}
        return _FakeAgent()
    deepagents_mod.create_deep_agent = _stub_create
    return deepagents_mod


if not _HAVE_DEEPAGENTS:
    sys.modules["deepagents"] = _make_stub_deepagents()


# Monkey-patch _init_model so agent tests work without API keys
def _stub_init_model():
    """Return a fake LangChain chat model for test environments."""
    # Install stub deepagents so create_deep_agent still works
    import deep_agent as _da
    da_mod = sys.modules.get("deepagents")
    if da_mod is None or not hasattr(da_mod, "create_deep_agent"):
        sys.modules["deepagents"] = _make_stub_deepagents()
    class _FakeModel:
        pass
    return _FakeModel()

# Force mock fallback (no CouchDB / AssetOpsBench needed)
os.environ.setdefault("ASSETOPS", "")
os.environ.setdefault("USE_IOT_SUBPROCESS", "0")
os.environ.setdefault("USE_WO_SUBPROCESS", "0")
os.environ.setdefault("USE_TSFM_SUBPROCESS", "0")
os.environ.setdefault("ENABLE_CONDITIONAL_DEEP_TSFM", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Patch deep_agent._init_model before any test imports SkillAgent / run_deep_agent
import deep_agent as _da  # noqa: E402
_da._init_model = _stub_init_model  # replace real LLM init with stub
# Also patch create_deep_agent so SkillAgent builds without a real model
import importlib
_deepagents = _make_stub_deepagents()
_da.create_deep_agent = _deepagents.create_deep_agent


# ── Tool-level unit tests (no LLM / no live backend required) ─────────────────

def test_iot_data_retrieval_tool():
    from deep_agent import iot_data_retrieval_tool
    raw = iot_data_retrieval_tool.invoke({"asset_id": "Chiller 6", "lookback_days": 7})
    data = json.loads(raw)
    assert "error" not in data, data
    assert "readings" in data
    assert data["asset_id"] == "Chiller 6"


def test_sensor_metadata_tool():
    from deep_agent import sensor_metadata_tool
    raw = sensor_metadata_tool.invoke({"asset_id": "Chiller 6"})
    data = json.loads(raw)
    assert "error" not in data, data
    assert "sensors" in data


def test_lightweight_anomaly_tool():
    from deep_agent import lightweight_anomaly_tool
    raw = lightweight_anomaly_tool.invoke({"asset_id": "Chiller 6", "lookback_days": 7})
    data = json.loads(raw)
    assert "error" not in data, data
    assert "anomalies_detected" in data
    assert "severity" in data


def test_fmsr_root_cause_tool():
    from deep_agent import lightweight_anomaly_tool, fmsr_root_cause_tool
    anomaly_raw = lightweight_anomaly_tool.invoke({"asset_id": "Chiller 6"})
    anomaly = json.loads(anomaly_raw)
    sensor_data = anomaly  # sensor_data embedded by lightweight_anomaly_tool
    raw = fmsr_root_cause_tool.invoke({
        "asset_id": "Chiller 6",
        "anomaly_json": anomaly_raw,
        "sensor_data_json": json.dumps(sensor_data),
        "task": "Why is Chiller 6 behaving abnormally?",
    })
    data = json.loads(raw)
    assert "error" not in data, data
    assert "failure" in data
    assert "diagnosis_confidence" in data
    assert "deep_tsfm_invoked" in data


def test_forecasting_tool():
    from deep_agent import forecasting_tool
    raw = forecasting_tool.invoke({
        "asset_id": "Chiller 9",
        "sensor": "condenser_flow_GPM",
        "horizon_days": 7,
    })
    data = json.loads(raw)
    assert "error" not in data, data
    assert "forecast" in data
    assert "maintenance_needed" in data


def test_work_order_tool():
    from deep_agent import work_order_tool
    raw = work_order_tool.invoke({
        "asset_id": "Chiller 6",
        "failure": "bearing_failure",
        "priority": "high",
    })
    data = json.loads(raw)
    assert "error" not in data, data
    assert "work_order_id" in data


def test_run_deep_agent_metadata_scenario():
    """Metadata scenario: only sensor_metadata_tool should be needed."""
    from deep_agent import run_deep_agent
    out = run_deep_agent("What sensors are available for Chiller 6, and what do they measure?")
    assert "result" in out
    assert "metrics" in out
    m = out["metrics"]
    assert isinstance(m["latency_s"], float)
    assert isinstance(m["tool_calls"], int)


def test_skill_agent_run():
    """SkillAgent.run returns the evaluator-compatible dict shape."""
    from deep_agent import SkillAgent
    agent = SkillAgent()
    out = agent.run("Was there any abnormal behavior in Chiller 9 over the past week?")
    assert "result" in out
    assert "metrics" in out
    m = out["metrics"]
    for key in ("plan", "tool_calls", "latency_s", "deep_tsfm_invoked",
                "diagnosis_confidence", "diagnosis_confidence_pre_deep"):
        assert key in m, f"Missing metrics key: {key}"


if __name__ == "__main__":
    tests = [
        test_iot_data_retrieval_tool,
        test_sensor_metadata_tool,
        test_lightweight_anomaly_tool,
        test_fmsr_root_cause_tool,
        test_forecasting_tool,
        test_work_order_tool,
        test_run_deep_agent_metadata_scenario,
        test_skill_agent_run,
    ]
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
