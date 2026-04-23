import json
from pathlib import Path

from trajectory_log import (
    append_trajectory_line,
    build_eval_trajectory,
    summarize_context,
)


def test_summarize_sensor_data_shapes():
    ctx = {
        "sensor_data": {
            "asset_id": "Chiller 6",
            "lookback_days": 7,
            "source": "mock",
            "readings": {"a": list(range(1000)), "b": [1.0, 2.0]},
        }
    }
    s = summarize_context(ctx)
    assert s["sensor_data"]["reading_shapes"] == {"a": 1000, "b": 2}
    assert "readings" not in s["sensor_data"]


def test_append_trajectory_line_writes_jsonl(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    append_trajectory_line({"x": 1}, path=str(p))
    append_trajectory_line({"x": 2}, path=str(p))
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["x"] == 1


def test_skill_agent_appends_trajectory_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAJECTORY_LOG_PATH", str(tmp_path / "run.jsonl"))
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from agent import SkillAgent

    SkillAgent().run("What sensors are available for Chiller 6?")
    text = (tmp_path / "run.jsonl").read_text(encoding="utf-8").strip()
    rec = json.loads(text.splitlines()[0])
    assert rec["kind"] == "skill_agent"
    assert rec["plan"]
    assert rec["skill_steps"]


def test_build_eval_trajectory_answer_preview():
    out = {
        "result": {"answer": "x" * 900},
        "metrics": {"plan": ["direct_llm"]},
    }
    r = build_eval_trajectory(
        condition="A",
        theta="",
        task_id="T01",
        category="x",
        task="hello",
        run_output=out,
    )
    assert "answer_preview" in r
    assert len(r["answer_preview"]) < 900
