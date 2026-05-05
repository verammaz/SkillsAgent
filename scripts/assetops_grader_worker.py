#!/usr/bin/env python3
"""Run AssetOps ``EvaluationAgent`` on a JSON payload (same logic as ``graders.evaluation_agent``).

**Do not run this file with plain ``python`` from SkillsAgent.** It must be
invoked with the ``scenario-server`` uv project (ReActXen on PYTHONPATH), e.g.::

    cd /path/to/AssetOpsBench/aobench
    uv run --directory scenario-server python \\
        /path/to/SkillsAgent/scripts/assetops_grader_worker.py /tmp/payload.json

Input file: JSON list of objects with keys
  ``task_id``, ``condition``, ``theta`` (optional), ``query``,
  ``characteristic_form``, ``result``, ``trace``, ``model_id`` (optional).

Stdout: JSON list of rows including ``overall_correct`` and the six metric
booleans. If the evaluator crashes, ``evaluator_error`` is set (the upstream
``evaluation_agent`` helper would only return a stub ``{"name":"result","value":False}``
detail with no explanation).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _slug(name: str) -> str:
    s = name.lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _grade_one(
    *,
    task_id: str,
    condition: str,
    theta: str,
    query: str,
    characteristic: str,
    actual: str,
    trace: str,
    model_id: int,
) -> dict:
    from reactxen.agents.evaluation_agent.agent import EvaluationAgent

    row: dict = {
        "task_id": task_id,
        "condition": condition,
        "theta": theta,
    }
    try:
        eval_agent = EvaluationAgent(model_id=model_id)
        review = eval_agent.evaluate_response(
            agent_response=actual,
            characteristic_answer=characteristic,
            question=query,
            agent_think=trace,
        )
    except Exception as exc:
        row["overall_correct"] = False
        row["evaluator_error"] = f"{type(exc).__name__}: {exc}"
        return row

    overall = bool(
        review.get("task_completion")
        and review.get("data_retrieval_accuracy")
        and review.get("generalized_result_verification")
        and review.get("agent_sequence_correct")
        and review.get("clarity_and_justification")
        and (review.get("hallucinations") is False)
    )
    row["overall_correct"] = overall
    details = [
        ("Task Completion", review.get("task_completion")),
        ("Data Retrieval Accuracy", review.get("data_retrieval_accuracy")),
        (
            "Generalized Result Verification",
            review.get("generalized_result_verification"),
        ),
        ("Agent Sequence Correct", review.get("agent_sequence_correct")),
        ("Clarity & Justification", review.get("clarity_and_justification")),
        ("Hallucinations", review.get("hallucinations")),
        ("Suggestions", review.get("suggestions", "No suggestions")),
    ]
    for name, val in details:
        row[_slug(name)] = val
    return row


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: assetops_grader_worker.py PAYLOAD.json", file=sys.stderr)
        sys.exit(2)
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    out: list[dict] = []
    for item in payload:
        out.append(
            _grade_one(
                task_id=str(item.get("task_id", "")),
                condition=str(item.get("condition", "")),
                theta=str(item.get("theta", "") or ""),
                query=str(item.get("query", "")),
                characteristic=str(item.get("characteristic_form", "")),
                actual=str(item.get("result", "") or ""),
                trace=str(item.get("trace", "") or ""),
                model_id=int(item.get("model_id", 16)),
            )
        )
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
