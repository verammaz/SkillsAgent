"""Run SkillsAgent ablations and score them with AssetOpsBench's evaluator.

This script bridges local ablation conditions (A/B/C/D/F/E_theta) to the
AssetOpsBench scenario-server grading API.

Requirements:
  1) Start scenario server (default URL http://localhost:8099)
  2) Ensure SkillsAgent env is configured (.env)

Example:
  python scripts/score_with_assetopsbench.py \
      --scenario-set 13aab653-66fe-4fe6-84d8-89f1b18eede3 \
      --conditions C D F E \
      --output-dir eval_results/aob_tsfm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_runner import (  # noqa: E402
    THETA_VALUES,
    run_condition_a,
    run_condition_b,
    run_condition_c,
    run_condition_d,
    run_condition_e,
    run_condition_f,
)


@dataclass
class ScenarioEntry:
    scenario_id: str
    query: str
    metadata: dict


def _http_json(url: str, method: str = "GET", body: dict | None = None) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}: {msg}") from exc


def fetch_scenarios(server_url: str, scenario_set_id: str, limit: int | None) -> list[ScenarioEntry]:
    url = f"{server_url.rstrip('/')}/scenario-set/{scenario_set_id}"
    payload = _http_json(url, method="GET")
    scenarios = payload.get("scenarios", []) or []
    out: list[ScenarioEntry] = []
    for s in scenarios:
        sid = str(s.get("id", ""))
        q = str(s.get("query", ""))
        if sid and q:
            out.append(ScenarioEntry(scenario_id=sid, query=q, metadata=s.get("metadata", {}) or {}))
    if limit is not None:
        out = out[: max(0, limit)]
    return out


def _build_submission_answer(run_output: dict, condition: str, theta: str) -> str:
    result_obj = run_output.get("result", {})
    metrics = run_output.get("metrics", {})
    # Evaluator expects a JSON-serialized string with {"result": ..., "trace": ...}
    trace = {
        "condition": condition,
        "theta": theta,
        "plan": metrics.get("plan"),
        "skills_executed": metrics.get("skills_executed"),
        "skills_skipped": metrics.get("skills_skipped"),
        "skill_steps": metrics.get("skill_steps"),
        "tool_calls": metrics.get("tool_calls"),
        "deep_tsfm_invoked": metrics.get("deep_tsfm_invoked"),
        "diagnosis_confidence": metrics.get("diagnosis_confidence"),
    }
    wrapped = {
        "result": json.dumps(result_obj, default=str),
        "trace": json.dumps(trace, default=str),
    }
    return json.dumps(wrapped, default=str)


def submit_for_grading(
    server_url: str,
    scenario_set_id: str,
    answers: list[dict[str, str]],
    tracking_context: dict | None = None,
) -> dict:
    url = f"{server_url.rstrip('/')}/scenario-set/{scenario_set_id}/grade"
    body = {"submission": answers}
    if tracking_context:
        body["tracking_context"] = tracking_context
    return _http_json(url, method="POST", body=body)


def _answer_from_eval_row(row: dict) -> dict[str, str]:
    sid = str(row.get("scenario_id") or row.get("task_id") or "")
    result_json = row.get("result_json") or "{}"
    trace_json = row.get("trace_json") or "{}"
    wrapped = {"result": result_json, "trace": trace_json}
    return {"scenario_id": sid, "answer": json.dumps(wrapped, default=str)}


def load_eval_csv_submissions(
    csv_path: str | Path,
    *,
    conditions: list[str] | None = None,
    theta_values: list[str] | None = None,
) -> tuple[str, dict[tuple[str, str], list[dict[str, str]]]]:
    """Group eval_runner CSV rows into grading submissions by (condition, theta)."""
    want_conditions = {c.upper() for c in (conditions or [])}
    want_thetas = {str(t) for t in (theta_values or [])}
    by_variant: dict[tuple[str, str], list[dict[str, str]]] = {}
    scenario_set_id = ""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("error"):
                continue
            cond = str(row.get("condition", ""))
            theta = str(row.get("theta", ""))
            if want_conditions and cond[:1].upper() not in want_conditions:
                continue
            if want_thetas and theta and theta not in want_thetas:
                continue
            sid = str(row.get("scenario_id") or row.get("task_id") or "").strip()
            if not sid:
                continue
            if row.get("scenario_set_id"):
                scenario_set_id = str(row["scenario_set_id"])
            key = (cond, theta)
            by_variant.setdefault(key, []).append(_answer_from_eval_row(row))
    return scenario_set_id, by_variant


def _condition_variants(selected: list[str], theta_values: list[str]) -> list[tuple[str, str, Callable[[str], dict]]]:
    funcs = {
        "A": run_condition_a,
        "B": run_condition_b,
        "C": run_condition_c,
        "D": run_condition_d,
        "F": run_condition_f,
    }
    variants: list[tuple[str, str, Callable[[str], dict]]] = []
    for c in selected:
        uc = c.upper()
        if uc == "E":
            for t in theta_values:
                variants.append((f"E_full_theta_{t.replace('.', '_')}", t, run_condition_e))
        elif uc in funcs:
            name = {
                "A": "A_raw_llm",
                "B": "B_tool_baseline",
                "C": "C_planning_only",
                "D": "D_skills_knowledge_no_deep_tsfm",
                "F": "F_skills_knowledge_always_deep",
            }[uc]
            variants.append((name, "", funcs[uc]))
    return variants


def main() -> None:
    load_dotenv(".env")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server-url", default=os.getenv("AOB_SCENARIO_SERVER", "http://localhost:8099"))
    ap.add_argument("--scenario-set", default="", help="Scenario-set UUID from /scenario-types.")
    ap.add_argument("--conditions", nargs="+", default=["C", "D", "F", "E"], help="Subset of A B C D F E.")
    ap.add_argument("--theta-values", nargs="+", default=list(THETA_VALUES), help="Used when conditions includes E.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of fetched scenarios.")
    ap.add_argument(
        "--from-eval-csv",
        default=None,
        metavar="PATH",
        help="Score existing eval_runner CSV rows instead of re-running agent.",
    )
    ap.add_argument("--output-dir", type=Path, default=Path("eval_results/aob_scored"))
    ns = ap.parse_args()

    ns.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    if ns.from_eval_csv:
        scenario_set_id_csv, by_variant = load_eval_csv_submissions(
            ns.from_eval_csv,
            conditions=ns.conditions,
            theta_values=ns.theta_values,
        )
        scenario_set_id = ns.scenario_set or scenario_set_id_csv
        if not scenario_set_id:
            raise RuntimeError("scenario_set_id missing: pass --scenario-set or include scenario_set_id column in CSV.")
        if not by_variant:
            raise RuntimeError("No matching rows in --from-eval-csv after filters.")

        for (cond_name, theta), answers in sorted(by_variant.items()):
            print(f"\nScoring from CSV: {cond_name} theta={theta or '-'} n={len(answers)}")
            graded = submit_for_grading(ns.server_url, scenario_set_id, answers)
            raw_path = ns.output_dir / f"graded_{cond_name}.json"
            raw_path.write_text(json.dumps(graded, indent=2) + "\n")
            grades = graded.get("grades", []) or []
            correct = sum(1 for g in grades if bool(g.get("correct")))
            n = len(grades) or 1
            summary_rows.append(
                {
                    "condition": cond_name,
                    "theta": theta,
                    "scenario_set_id": scenario_set_id,
                    "scored": len(grades),
                    "correct_count": correct,
                    "accuracy": round(correct / n, 4),
                    "raw_json": str(raw_path),
                }
            )
            for g in grades:
                detail_rows.append(
                    {
                        "condition": cond_name,
                        "theta": theta,
                        "scenario_id": str(g.get("scenario_id")),
                        "correct": bool(g.get("correct")),
                        "details": json.dumps(g.get("details", []), default=str),
                        "tool_calls": "",
                        "total_cost": "",
                        "latency_s": "",
                        "deep_tsfm_invoked": "",
                        "diagnosis_confidence": "",
                    }
                )
    else:
        if not ns.scenario_set:
            raise RuntimeError("--scenario-set is required unless --from-eval-csv is used")
        scenarios = fetch_scenarios(ns.server_url, ns.scenario_set, ns.limit)
        if not scenarios:
            raise RuntimeError("No scenarios fetched from server.")

        variants = _condition_variants(ns.conditions, ns.theta_values)
        if not variants:
            raise RuntimeError("No condition variants selected.")

        for cond_name, theta, run_fn in variants:
            print(f"\nCondition: {cond_name}  theta={theta or '-'}")
            answers: list[dict[str, str]] = []
            run_outputs: dict[str, dict] = {}
            for i, sc in enumerate(scenarios, start=1):
                if theta:
                    os.environ["RCA_CONFIDENCE_THETA"] = theta
                out = run_fn(sc.query)
                run_outputs[sc.scenario_id] = out
                answers.append(
                    {
                        "scenario_id": sc.scenario_id,
                        "answer": _build_submission_answer(out, cond_name, theta),
                    }
                )
                print(f"  [{i:>3}/{len(scenarios)}] {sc.scenario_id}")

            graded = submit_for_grading(ns.server_url, ns.scenario_set, answers)
            raw_path = ns.output_dir / f"graded_{cond_name}.json"
            raw_path.write_text(json.dumps(graded, indent=2) + "\n")

            grades = graded.get("grades", []) or []
            correct = sum(1 for g in grades if bool(g.get("correct")))
            n = len(grades) or 1
            summary_rows.append(
                {
                    "condition": cond_name,
                    "theta": theta,
                    "scenario_set_id": ns.scenario_set,
                    "scored": len(grades),
                    "correct_count": correct,
                    "accuracy": round(correct / n, 4),
                    "raw_json": str(raw_path),
                }
            )

            for g in grades:
                sid = str(g.get("scenario_id"))
                out = run_outputs.get(sid, {})
                m = out.get("metrics", {})
                detail_rows.append(
                    {
                        "condition": cond_name,
                        "theta": theta,
                        "scenario_id": sid,
                        "correct": bool(g.get("correct")),
                        "details": json.dumps(g.get("details", []), default=str),
                        "tool_calls": m.get("tool_calls"),
                        "total_cost": m.get("total_cost"),
                        "latency_s": m.get("latency_s"),
                        "deep_tsfm_invoked": m.get("deep_tsfm_invoked"),
                        "diagnosis_confidence": m.get("diagnosis_confidence"),
                    }
                )

    summary_csv = ns.output_dir / "grading_summary.csv"
    details_csv = ns.output_dir / "grading_details.csv"
    with open(summary_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with open(details_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    print(f"\nWrote {summary_csv}")
    print(f"Wrote {details_csv}")


if __name__ == "__main__":
    main()

