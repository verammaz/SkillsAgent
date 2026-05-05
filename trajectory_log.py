"""Append-only JSONL trajectories for Phase 1 / 6 (proposal) and ablation analysis.

Set ``TRAJECTORY_LOG_PATH`` to a file path to log each ``SkillAgent.run()`` and
each eval row (conditions A–E). Large fields (sensor series) are summarized, not
dumped in full.

Optional: ``evaluate_all(..., trajectory_log_path=...)`` sets the log path for that run.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _truncate_str(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _truncate_value(v: Any, depth: int, max_str: int) -> Any:
    if depth <= 0:
        return "…"
    if isinstance(v, dict):
        return {k: _truncate_value(x, depth - 1, max_str) for k, x in list(v.items())[:40]}
    if isinstance(v, list):
        if len(v) > 24:
            return [_truncate_value(x, depth - 1, max_str) for x in v[:12]] + ["…"] + [
                _truncate_value(x, depth - 1, max_str) for x in v[-8:]
            ]
        return [_truncate_value(x, depth - 1, max_str) for x in v]
    if isinstance(v, str):
        return _truncate_str(v, max_str)
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return str(type(v).__name__)


def summarize_context(context: dict, *, max_str: int = 400) -> dict:
    """JSON-safe summary: reading shapes instead of full time series."""
    out: dict[str, Any] = {}
    for k, v in context.items():
        if k == "sensor_data" and isinstance(v, dict):
            rd = v.get("readings") or {}
            out[k] = {
                "asset_id": v.get("asset_id"),
                "lookback_days": v.get("lookback_days"),
                "source": v.get("source"),
                "reading_shapes": {
                    col: len(s) if isinstance(s, list) else type(s).__name__
                    for col, s in rd.items()
                },
            }
        elif k == "forecast" and isinstance(v, dict):
            fc = {**v}
            arr = fc.get("forecasted")
            if isinstance(arr, list) and len(arr) > 32:
                fc["forecasted"] = arr[:5] + [f"…({len(arr)} values)…"] + arr[-5:]
            out[k] = fc
        elif isinstance(v, dict):
            out[k] = _truncate_value(v, depth=5, max_str=max_str)
        elif isinstance(v, list):
            out[k] = _truncate_value(v, depth=4, max_str=max_str)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v if not isinstance(v, str) else _truncate_str(v, max_str)
        else:
            out[k] = type(v).__name__
    return out


def append_trajectory_line(
    record: dict,
    path: str | Path | None = None,
) -> None:
    """Append one JSON object as a line. ``path`` defaults to ``TRAJECTORY_LOG_PATH``."""
    p = path or os.getenv("TRAJECTORY_LOG_PATH")
    if not p:
        return
    fp = Path(p)
    fp.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with open(fp, "a", encoding="utf-8") as fh:
        fh.write(line)


def build_agent_trajectory(
    *,
    task: str,
    asset_id: str,
    plan: list,
    metrics: dict,
    context: dict,
    skill_steps: list[dict],
) -> dict:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": "skill_agent",
        "task": task,
        "asset_id": asset_id,
        "plan": plan,
        "metrics": metrics,
        "context_summary": summarize_context(context),
        "skill_steps": skill_steps,
    }


def _grader_result_string(ctx: Any) -> str:
    """TSFM scenario server passes ``unwrap['result']`` to EvaluationAgent as ``agent_response``."""
    if isinstance(ctx, dict):
        ans = ctx.get("answer")
        if isinstance(ans, str) and ans.strip():
            return ans
        return json.dumps(ctx, default=str)
    if ctx in ("", None):
        return ""
    return json.dumps(ctx, default=str)


def _metrics_for_grader_trace(m: dict) -> dict:
    """Execution trace for ``unwrap['trace']`` (EvaluationAgent ``agent_think``) — no huge blobs."""
    if not isinstance(m, dict):
        return {}
    keys = (
        "plan",
        "tool_calls",
        "skills_executed",
        "skills_skipped",
        "skipped_conditional",
        "skipped_early_stop",
        "stopped_at",
        "total_cost",
        "latency_s",
        "diagnosis_confidence",
        "diagnosis_confidence_pre_deep",
        "deep_tsfm_invoked",
    )
    out: dict[str, Any] = {k: m[k] for k in keys if k in m}
    steps = m.get("skill_steps")
    if isinstance(steps, list) and steps:
        brief: list[Any] = []
        for s in steps[:48]:
            if isinstance(s, dict):
                brief.append(
                    {
                        k: s.get(k)
                        for k in (
                            "skill",
                            "status",
                            "should_stop",
                            "latency_s",
                            "output_keys",
                            "deep_tsfm_invoked",
                            "cost_charged",
                            "tool_calls",
                        )
                        if k in s
                    }
                )
            else:
                brief.append(s)
        if len(steps) > 48:
            brief.append({"_truncated": f"{len(steps)} skill_steps total"})
        out["skill_steps"] = brief
    elif steps is not None:
        out["skill_steps"] = steps
    return out


def build_eval_trajectory(
    *,
    condition: str,
    theta: str,
    task_id: str,
    category: str,
    task: str,
    run_output: dict,
) -> dict:
    """Flatten eval_runner row + agent output into one JSONL record.

    The AssetOps TSFM scenario handler grades ``ScenarioAnswer.answer`` after
    ``json.loads`` — it must be a JSON object with **string** keys ``result`` and
    ``trace`` (see ``aob_tsfm.AOBTSFMScenarios._grade_answer``). We expose the same
    payload as ``submission_answer_json`` so you can submit it without hand-merging
    from ``metrics`` / ``context_summary``.

    **Important:** the server-side ``EvaluationAgent`` also scores
    ``agent_sequence_correct`` against real TSFM / MCP-style tool traces. SkillsAgent
    ``plan`` entries (e.g. ``direct_llm``, ``data_retrieval``) will not match that
    rubric even when ``submission_answer_json`` is well-formed — expect low or zero
    official pass rate unless you emit bench-shaped traces.
    """
    m = run_output.get("metrics") or {}
    ctx = run_output.get("result") or {}
    base = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": "eval_ablation",
        "condition": condition,
        "theta": theta,
        "task_id": task_id,
        "category": category,
        "task": task,
        "metrics": m,
        "context_summary": summarize_context(ctx) if isinstance(ctx, dict) else {},
    }
    if isinstance(ctx, dict) and "answer" in ctx:
        base["answer_preview"] = _truncate_str(str(ctx.get("answer", "")), 800)

    res_str = _grader_result_string(ctx)
    trace_obj = _metrics_for_grader_trace(m)
    trace_str = json.dumps(trace_obj, default=str)
    max_r, max_t = 120_000, 80_000
    if len(res_str) > max_r:
        res_str = res_str[: max_r - 1] + "…"
    if len(trace_str) > max_t:
        trace_str = trace_str[: max_t - 1] + "…"
    base["submission_answer_json"] = json.dumps(
        {"result": res_str, "trace": trace_str},
        ensure_ascii=False,
    )
    return base
