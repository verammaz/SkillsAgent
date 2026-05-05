"""Ablation runner (conditions A–F + E θ sweep).

  A. Raw LLM — no tools
  B. Tool baseline — static IoT → TSFM-lite → FMSR → WO chain (ReAct stand-in)
  C. Planning only — planner + skills, knowledge disabled, no skipping
  D. Skills + knowledge — full SkillAgent, deep TSFM disabled
  F. Skills + knowledge — always run deep TSFM in RCA (no θ gate), no cost budget
  E. Full system — θ-gated deep TSFM + default cost budget (swept per ``THETA_VALUES``)

Condition E sweeps ``RCA_CONFIDENCE_THETA`` over a range that straddles the
observed graded-confidence knee (~0.6–0.65 on the default task bank) so the
cost/accuracy frontier exposes both deep-off and deep-on regimes.

Usage:
    python eval_runner.py [--output-dir DIR] [--hf-limit N] [--trajectory-log PATH]
                          [--tsfm-report PATH] [--prepend-builtin]
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

import dotenv

dotenv.load_dotenv()

from scenario_loader import BUILTIN_TASK_BANK

TASK_BANK = BUILTIN_TASK_BANK


def _normalize_category(raw: str, task: str) -> str:
    """Map scenario metadata/query into eval categories expected by this runner."""
    v = (raw or "").strip().lower()
    if v in {"fault_diagnosis", "fault-diagnosis", "fault diagnosis", "workorder", "workorders", "fmsa"}:
        return "fault_diagnosis"
    if v in {"anomaly_detection", "anomaly-detection", "anomaly detection", "tsad", "anomaly"}:
        return "anomaly_detection"
    if v in {"forecasting", "forecast"}:
        return "forecasting"
    if v in {"metadata", "sensor_metadata", "sensor-metadata"}:
        return "metadata"

    t = task.lower()
    if any(w in t for w in ("what sensor", "metadata", "measure", "unit", "available")):
        return "metadata"
    if any(w in t for w in ("forecast", "predict", "next week", "future")):
        return "forecasting"
    if any(w in t for w in ("anomal", "abnormal", "unusual")):
        return "anomaly_detection"
    return "fault_diagnosis"


def load_scenario_file(path: str | Path) -> list[tuple[str, str, str]]:
    """Load scenario snapshot JSON/JSONL exported from AssetOpsBench server.

    Expected fields per row:
      - ``scenario_id`` or ``id``
      - ``query`` or ``text``
      - optional ``category`` and/or ``metadata``
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"scenario file not found: {p}")

    def row_to_tuple(row: dict[str, Any], idx: int) -> tuple[str, str, str] | None:
        sid = str(row.get("scenario_id") or row.get("id") or f"S_{idx}")
        task = str(row.get("query") or row.get("text") or "").strip()
        if not task:
            return None
        raw_cat = row.get("category")
        if not raw_cat and isinstance(row.get("metadata"), dict):
            md = row["metadata"]
            raw_cat = md.get("category") or md.get("type") or md.get("domain")
        cat = _normalize_category(str(raw_cat or ""), task)
        return sid, task, cat

    if p.suffix.lower() == ".jsonl":
        rows: list[tuple[str, str, str]] = []
        with open(p, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict):
                    item = row_to_tuple(rec, i)
                    if item:
                        rows.append(item)
        return rows

    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # exported payload shape: {"scenario_set_id": ..., "scenarios": [...]}
        recs = data.get("scenarios", [])
    else:
        recs = data
    rows = []
    for i, rec in enumerate(recs):
        if isinstance(rec, dict):
            item = row_to_tuple(rec, i)
            if item:
                rows.append(item)
    return rows


def run_condition_a(task: str) -> dict:
    t0 = time.time()
    answer = ""
    system = (
        "You are an industrial asset operations agent. "
        "Answer the following question about equipment maintenance."
    )
    try:
        # Use the shared, provider-agnostic helper so Condition A picks up the
        # same retry/fallback logic as the rest of the agent (watsonx model
        # fallbacks, then gemini/anthropic/groq).
        from skills import _call_llm

        answer = _call_llm(system, f"Task: {task}", max_tokens=256).strip()
        if not answer:
            answer = "error: all LLM providers returned empty"
    except Exception as e:
        answer = f"error: {e}"

    return {
        "task": task,
        "plan": ["direct_llm"],
        "result": {"answer": answer},
        "metrics": {
            "plan": ["direct_llm"],
            "tool_calls": 1,
            "skills_skipped": [],
            "skipped_conditional": [],
            "skipped_early_stop": [],
            "total_cost": 0.0,
            "latency_s": round(time.time() - t0, 3),
            "diagnosis_confidence": None,
            "diagnosis_confidence_pre_deep": None,
            "deep_tsfm_invoked": False,
        },
    }


def run_condition_b(task: str) -> dict:
    """Unconditional tool pipeline — no skill registry (ReAct stand-in)."""
    from agent import _extract_asset
    from tools import (
        detect_anomaly,
        forecast_sensor,
        generate_work_order,
        get_asset_metadata,
        get_sensor_data,
        map_failure,
    )

    asset_id = _extract_asset(task)
    t0 = time.time()
    context: dict = {}
    calls = 0
    t = task.lower()

    if any(w in t for w in ("what sensor", "metadata", "available")):
        context["metadata"] = get_asset_metadata(asset_id)
        calls = 1
    elif any(w in t for w in ("forecast", "predict", "next week", "future")):
        data = get_sensor_data(asset_id)
        calls += 1
        forecast = forecast_sensor(
            asset_id,
            "condenser_flow_GPM",
            horizon_days=7,
            sensor_data=data,
            task=task,
        )
        calls += 1
        context["sensor_data"] = data
        context["forecast"] = forecast
    else:
        data = get_sensor_data(asset_id)
        calls += 1
        anomaly = detect_anomaly(data)
        calls += 1
        failure = map_failure(anomaly, None, asset_id)
        calls += 1
        context["work_order"] = generate_work_order(asset_id, failure, "high")
        calls += 1
        context.update({"sensor_data": data, "anomaly_analysis": anomaly, "failure": failure})

    return {
        "task": task,
        "plan": ["B_static_tools"],
        "result": context,
        "metrics": {
            "plan": ["B_static_tools"],
            "tool_calls": calls,
            "skills_skipped": [],
            "skipped_conditional": [],
            "skipped_early_stop": [],
            "total_cost": round(calls * 0.25, 3),
            "latency_s": round(time.time() - t0, 3),
            "diagnosis_confidence": None,
            "diagnosis_confidence_pre_deep": None,
            "deep_tsfm_invoked": False,
        },
    }


def run_condition_c(task: str) -> dict:
    """Planner + skills; run every skill in plan (no should_skip, no knowledge).

    Matches proposal Table~5: condition C disables knowledge injection and deep
    TSFM gating (E adds both).
    """
    from agent import SkillAgent, _extract_asset
    from skills import SKILL_REGISTRY

    # No confidence-based deep TSFM inside RCA (that is introduced in cond.~E).
    with _env_override("ENABLE_CONDITIONAL_DEEP_TSFM", "0"), _env_override(
        "KNOWLEDGE_INJECTION", "0"
    ):
        agent = SkillAgent()
        asset_id = _extract_asset(task)
        plan = agent.plan(task)
        context = {}
        cost = 0.0
        calls = 0
        t0 = time.time()

        for skill_name in plan:
            skill = SKILL_REGISTRY.get(skill_name)
            if not skill:
                continue
            try:
                result = skill["fn"](asset_id, context=context, task="")
                context.update(result.get("output", {}))
                cost += skill["cost"]
                calls += 1
            except Exception:
                pass

        return {
            "task": task,
            "plan": plan,
            "result": context,
            "metrics": {
                "plan": plan,
                "tool_calls": calls,
                "skills_skipped": [],
                "skipped_conditional": [],
                "skipped_early_stop": [],
                "total_cost": round(cost, 3),
                "latency_s": round(time.time() - t0, 3),
                "diagnosis_confidence": context.get("diagnosis_confidence"),
                "diagnosis_confidence_pre_deep": context.get("diagnosis_confidence_pre_deep"),
                "deep_tsfm_invoked": bool(context.get("deep_tsfm_invoked", False)),
            },
        }


def run_condition_d(task: str) -> dict:
    """Full SkillAgent with knowledge; disable deep TSFM gating (proposal cond.~D)."""
    with _env_override("ENABLE_CONDITIONAL_DEEP_TSFM", "0"):
        from agent import SkillAgent

        return SkillAgent().run(task)


def _cost_budget_for_condition_e() -> float | None:
    """Derive the Condition~E cost budget.

    Resolution order:
      1. ``COST_BUDGET`` env var — explicit override (float or ``none``).
      2. 80% of the full plan cost summed from :data:`SKILL_REGISTRY` + ``DEEP_TSFM_COST`` —
         tight enough that at least one low-priority skill (e.g. ``validate_failure``
         or ``metadata_retrieval``) gets skipped on heavy plans.
    """
    raw = os.getenv("COST_BUDGET")
    if raw is not None:
        if raw.strip().lower() in ("", "none", "off", "0"):
            return None
        try:
            return float(raw)
        except ValueError:
            pass

    from agent import DEEP_TSFM_COST
    from skills import SKILL_REGISTRY

    full_plan_cost = sum(m["cost"] for m in SKILL_REGISTRY.values()) + DEEP_TSFM_COST
    return round(full_plan_cost * 0.8, 3)


def run_condition_f(task: str) -> dict:
    """Skills + knowledge; deep TSFM in RCA on every fault path (no θ gate).

    Compare to condition E: same planner/skills/knowledge, but ``deep_tsfm_refine_anomalies``
    always runs when conditional deep TSFM is enabled (``RCA_ALWAYS_DEEP_TSFM=1``),
    and no cost budget so skills are not skipped for budgeting reasons.
    """
    from agent import SkillAgent

    with _env_override("ENABLE_CONDITIONAL_DEEP_TSFM", "1"):
        with _env_override("RCA_ALWAYS_DEEP_TSFM", "1"):
            with _env_override("COST_BUDGET", "none"):
                return SkillAgent(cost_budget=None).run(task)


def run_condition_e(task: str) -> dict:
    """Full system including conditional deep TSFM + cost-aware skipping (cond.~E)."""
    from agent import SkillAgent

    budget = _cost_budget_for_condition_e()
    with _env_override("ENABLE_CONDITIONAL_DEEP_TSFM", "1"):
        with _env_override("RCA_ALWAYS_DEEP_TSFM", "0"):
            return SkillAgent(cost_budget=budget).run(task)


@contextmanager
def _env_override(key: str, value: str):
    prev = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


THETA_VALUES = ("0.5", "0.6", "0.65", "0.7", "0.8", "0.9", "0.95")


def _wandb_log_last_row(rows: list) -> None:
    if not rows:
        return
    try:
        from wandb_tracking import wandb_eval_log_row

        wandb_eval_log_row(rows[-1])
    except Exception:
        pass


def evaluate_all(
    output_dir: str = "eval_results",
    *,
    task_bank: list[tuple[str, str, str]] | None = None,
    trajectory_log_path: str | None = None,
    condition_codes: list[str] | None = None,
    theta_values: list[str] | None = None,
    scenario_set_id: str = "",
) -> None:
    """Run ablations. Pass ``task_bank=load_hf_scenario_tasks(limit=...)`` for HF scenarios.

    If ``trajectory_log_path`` is set, append one JSON object per task × condition
    (summarized context; see ``trajectory_log``).
    """
    tasks = task_bank if task_bank is not None else TASK_BANK
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    selected = {c.upper() for c in (condition_codes or ["A", "B", "C", "D", "F", "E"])}
    from wandb_tracking import wandb_eval_finish, wandb_eval_init

    _theta_for_cfg = list(theta_values or list(THETA_VALUES)) if "E" in selected else []
    wandb_eval_init(
        {
            "n_tasks": len(tasks),
            "condition_codes": sorted(selected),
            "theta_values": _theta_for_cfg,
            "scenario_set_id": scenario_set_id or None,
            "output_dir": str(Path(output_dir).resolve()),
            "tsfm_report_csv": (os.environ.get("TSFM_REPORT_CSV") or "").strip() or None,
        }
    )
    static_conditions = []
    if "A" in selected:
        static_conditions.append(("A_raw_llm", run_condition_a))
    if "B" in selected:
        static_conditions.append(("B_tool_baseline", run_condition_b))
    if "C" in selected:
        static_conditions.append(("C_planning_only", run_condition_c))
    if "D" in selected:
        static_conditions.append(("D_skills_knowledge_no_deep_tsfm", run_condition_d))
    if "F" in selected:
        static_conditions.append(("F_skills_knowledge_always_deep", run_condition_f))

    try:
        for cond_name, run_fn in static_conditions:
            print(f"\n{'─' * 60}\nCondition: {cond_name}\n{'─' * 60}")
            for task_id, task, category in tasks:
                print(f"  {task_id} [{category}] {task[:55]}...")
                _append_row(
                    rows,
                    cond_name,
                    "",
                    task_id,
                    category,
                    task,
                    run_fn,
                    scenario_set_id=scenario_set_id,
                    trajectory_log_path=trajectory_log_path,
                )

        if "E" in selected:
            for theta in (theta_values or list(THETA_VALUES)):
                cond_name = f"E_full_theta_{theta.replace('.', '_')}"
                print(f"\n{'─' * 60}\nCondition: {cond_name} (RCA_CONFIDENCE_THETA={theta})\n{'─' * 60}")
                with _env_override("RCA_CONFIDENCE_THETA", theta), _env_override(
                    "RCA_ALWAYS_DEEP_TSFM", "0"
                ):
                    for task_id, task, category in tasks:
                        print(f"  {task_id} [{category}] {task[:55]}...")
                        _append_row(
                            rows,
                            cond_name,
                            theta,
                            task_id,
                            category,
                            task,
                            run_condition_e,
                            scenario_set_id=scenario_set_id,
                            trajectory_log_path=trajectory_log_path,
                        )

        csv_path = Path(output_dir) / "ablation_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nDone. Results -> {csv_path}")
        _print_summary(rows)
    finally:
        wandb_eval_finish(Path(output_dir) / "ablation_results.csv")


_EXPECTED_SKILLS = {
    "fault_diagnosis":   {"root_cause_analysis", "work_order_generation"},
    "anomaly_detection": {"anomaly_detection", "root_cause_analysis"},
    "forecasting":       {"forecasting"},
    "metadata":          {"metadata_retrieval"},
}

_FINAL_KEYS = {
    "fault_diagnosis":   ("work_order", "failure"),
    "anomaly_detection": ("anomaly_analysis",),
    "forecasting":       ("forecast",),
    "metadata":          ("metadata",),
}


def _task_completion_score(
    category: str,
    task: str,
    metrics: dict,
    result: dict,
) -> float:
    """Heuristic 0–1 completion score used by the ablation CSV.

    Combines three signals, each equally weighted (1/3):

      1. **Plan coverage** — fraction of category-expected skills that ended up
         in the planner's ordered plan (sub-plans of the expected set count too,
         so a tighter plan is not penalized if the essential skills are present).
      2. **Executed coverage** — same, but over the skills that actually ran
         (executor may skip on conditional / cost-budget rules).
      3. **Final artefact** — does the result dict carry the category-specific
         deliverable (``work_order`` for fault_diagnosis with high severity,
         ``forecast`` for forecasting, etc.)?

    For Condition A (raw LLM, plan=[direct_llm]) and Condition B (static tool
    chain), plan coverage collapses to whether the LLM/tool pipeline produced
    the final artefact; we keep the scoring identical across conditions so the
    metric is comparable.
    """
    expected_skills = _EXPECTED_SKILLS.get(category, set())
    expected_keys = _FINAL_KEYS.get(category, ())

    plan = metrics.get("plan", []) or []
    executed_plan = [s for s in plan if s not in metrics.get("skills_skipped", [])]

    if expected_skills:
        plan_cov = len(expected_skills & set(plan)) / len(expected_skills)
        exec_cov = len(expected_skills & set(executed_plan)) / len(expected_skills)
    else:
        plan_cov = 1.0
        exec_cov = 1.0

    # ``result`` is Condition C/D/E's context dict, or A's {answer} / B's context.
    res = result if isinstance(result, dict) else {}
    artefact = 0.0
    if expected_keys:
        present = sum(1 for k in expected_keys if res.get(k))
        if present == 0 and "answer" in res and isinstance(res["answer"], str):
            # Condition A fallback: credit if the free-text answer mentions
            # the required artefact keyword (e.g. "work order", "forecast").
            ans = res["answer"].lower()
            kw = {
                "fault_diagnosis":   ("work order", "schedule"),
                "anomaly_detection": ("anomal", "abnormal"),
                "forecasting":       ("forecast", "predict"),
                "metadata":          ("sensor", "measure"),
            }.get(category, ())
            if kw and any(k in ans for k in kw):
                present = 1
        artefact = min(1.0, present / len(expected_keys))
    else:
        artefact = 1.0

    score = (plan_cov + exec_cov + artefact) / 3.0
    return round(score, 3)


def _append_row(
    rows: list,
    cond_name: str,
    theta: str,
    task_id: str,
    category: str,
    task: str,
    run_fn,
    *,
    scenario_set_id: str = "",
    trajectory_log_path: str | None = None,
) -> None:
    try:
        out = run_fn(task)
        m = out.get("metrics", {})
        result_obj = out.get("result", {})
        trace_obj = {
            "condition": cond_name,
            "theta": theta,
            "plan": m.get("plan"),
            "skills_executed": m.get("skills_executed"),
            "skills_skipped": m.get("skills_skipped"),
            "skill_steps": m.get("skill_steps"),
            "tool_calls": m.get("tool_calls"),
            "deep_tsfm_invoked": m.get("deep_tsfm_invoked"),
            "diagnosis_confidence": m.get("diagnosis_confidence"),
        }
        completion = _task_completion_score(category, task, m, out.get("result", {}))
        if trajectory_log_path:
            from trajectory_log import append_trajectory_line, build_eval_trajectory

            append_trajectory_line(
                build_eval_trajectory(
                    condition=cond_name,
                    theta=theta,
                    task_id=task_id,
                    category=category,
                    task=task,
                    run_output=out,
                ),
                path=trajectory_log_path,
            )
        rows.append(
            {
                "scenario_set_id": scenario_set_id,
                "condition": cond_name,
                "theta": theta,
                "task_id": task_id,
                "scenario_id": task_id,
                "category": category,
                "task": task[:80],
                "plan": json.dumps(m.get("plan", [])),
                "result_json": json.dumps(result_obj, default=str),
                "trace_json": json.dumps(trace_obj, default=str),
                "tool_calls": m.get("tool_calls", -1),
                "skipped_conditional": len(m.get("skipped_conditional", [])),
                "skipped_early_stop": len(m.get("skipped_early_stop", [])),
                "skills_skipped": len(m.get("skills_skipped", [])),
                "total_cost": m.get("total_cost", -1),
                "latency_s": m.get("latency_s", -1),
                "diagnosis_confidence": m.get("diagnosis_confidence", ""),
                "diagnosis_confidence_pre_deep": m.get("diagnosis_confidence_pre_deep", ""),
                "deep_tsfm_invoked": m.get("deep_tsfm_invoked", ""),
                "task_completion": completion,
                "error": "",
            }
        )
        _wandb_log_last_row(rows)
        print(
            f"    ok  calls={m.get('tool_calls')} "
            f"deep_tsfm={m.get('deep_tsfm_invoked')} "
            f"cost={m.get('total_cost')} lat={m.get('latency_s')}s"
        )
    except Exception as e:
        logger.error("    fail %s: %s", task_id, e)
        rows.append(
            {
                "scenario_set_id": scenario_set_id,
                "condition": cond_name,
                "theta": theta,
                "task_id": task_id,
                "scenario_id": task_id,
                "category": category,
                "task": task[:80],
                "plan": "",
                "result_json": "",
                "trace_json": "",
                "tool_calls": -1,
                "skipped_conditional": -1,
                "skipped_early_stop": -1,
                "skills_skipped": -1,
                "total_cost": -1,
                "latency_s": -1,
                "diagnosis_confidence": "",
                "diagnosis_confidence_pre_deep": "",
                "deep_tsfm_invoked": "",
                "task_completion": "",
                "error": str(e),
            }
        )
        _wandb_log_last_row(rows)


FIELDS = [
    "scenario_set_id",
    "condition",
    "theta",
    "task_id",
    "scenario_id",
    "category",
    "task",
    "plan",
    "result_json",
    "trace_json",
    "tool_calls",
    "skipped_conditional",
    "skipped_early_stop",
    "skills_skipped",
    "total_cost",
    "latency_s",
    "diagnosis_confidence",
    "diagnosis_confidence_pre_deep",
    "deep_tsfm_invoked",
    "task_completion",
    "error",
]


def _print_summary(rows: list) -> None:
    from collections import defaultdict

    by_cond = defaultdict(list)
    for r in rows:
        if isinstance(r["tool_calls"], int) and r["tool_calls"] >= 0:
            by_cond[r["condition"]].append(r)

    print(f"\n{'Condition':<38} {'n':>4} {'avg_calls':>10} {'avg_cost':>10} {'avg_lat(s)':>12}")
    for cond, task_rows in sorted(by_cond.items()):
        n = len(task_rows)
        avg = lambda k: sum(float(r[k]) for r in task_rows) / n
        print(f"{cond:<38} {n:>4} {avg('tool_calls'):>10.1f} {avg('total_cost'):>10.3f} {avg('latency_s'):>12.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation conditions A–F (+E theta sweep) on a task bank.")
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory for ablation_results.csv",
    )
    parser.add_argument(
        "--hf-limit",
        type=int,
        default=None,
        metavar="N",
        help="If set, load up to N scenarios from Hugging Face (falls back to builtin on error).",
    )
    parser.add_argument(
        "--trajectory-log",
        default=None,
        metavar="PATH",
        help="Append JSONL trajectory records (one per task × condition) to this file.",
    )
    parser.add_argument(
        "--tsfm-report",
        default=None,
        metavar="PATH",
        help=(
            "Load scenarios from tsfm_report.csv (id,type,label,text,tsfm_tools_called). "
            "If omitted but TSFM_REPORT_CSV is set, that path is used."
        ),
    )
    parser.add_argument(
        "--prepend-builtin",
        action="store_true",
        help="Prefix BUILTIN_TASK_BANK before tsfm-report rows (dedupe by task_id).",
    )
    parser.add_argument(
        "--scenario-file",
        default=None,
        metavar="PATH",
        help="Scenario snapshot JSON/JSONL (exported from AssetOpsBench scenario server).",
    )
    parser.add_argument(
        "--scenario-set-id",
        default="",
        metavar="UUID",
        help="Optional scenario-set UUID to carry through output CSV rows.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["A", "B", "C", "D", "F", "E"],
        help="Subset of conditions to run (A B C D F E).",
    )
    parser.add_argument(
        "--theta-values",
        nargs="+",
        default=list(THETA_VALUES),
        help="Theta values used when conditions include E.",
    )
    args = parser.parse_args()
    task_bank = None
    if args.scenario_file:
        task_bank = load_scenario_file(args.scenario_file)
    elif args.hf_limit is not None:
        from scenario_loader import load_hf_scenario_tasks

        task_bank = load_hf_scenario_tasks(limit=args.hf_limit)
    elif args.tsfm_report or os.getenv("TSFM_REPORT_CSV"):
        from scenario_loader import (
            BUILTIN_TASK_BANK,
            load_tsfm_report_tasks,
            default_tsfm_report_path,
        )

        csv_path = args.tsfm_report or os.getenv("TSFM_REPORT_CSV") or default_tsfm_report_path()
        if csv_path:
            extra = load_tsfm_report_tasks(csv_path)
            if args.prepend_builtin:
                seen = {tid for tid, _, _ in extra}
                prefix = [(tid, txt, cat) for tid, txt, cat in BUILTIN_TASK_BANK if tid not in seen]
                task_bank = prefix + extra
            else:
                task_bank = extra or None
        if task_bank is None:
            logger.warning(
                "No tsfm_report rows loaded — falling back to BUILTIN_TASK_BANK"
            )
    evaluate_all(
        args.output_dir,
        task_bank=task_bank,
        trajectory_log_path=args.trajectory_log,
        condition_codes=args.conditions,
        theta_values=args.theta_values,
        scenario_set_id=args.scenario_set_id,
    )
