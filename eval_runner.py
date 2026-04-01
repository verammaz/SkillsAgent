"""eval_runner.py

Four-condition ablation study for AssetOpsBench, matching the evaluation plan
from the proposal (Section 4):

  A. baseline          — direct LLM call, no skills, no planner
  B. +planning         — planner generates skill list, but no conditional exec
  C. +planning+skills  — planner + skills, no knowledge injection
  D. full system       — planner + skills + knowledge plugins (our approach)

Metrics collected per task:
  tool_calls, skills_skipped, total_cost, latency_s
  task_completion: fill in manually or wire to AssetOpsBench evaluator

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python eval_runner.py
    # results → eval_results/ablation_results.csv
"""

import csv
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)  # quiet during eval

import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file

# ── Task bank ────────────────────────────────────────────────────────────────
# Extend with the full AssetOpsBench task set

TASK_BANK = [
    ("T01", "Why is Chiller 6 behaving abnormally and do we need a work order?",    "fault_diagnosis"),
    ("T02", "Forecast next week's condenser water flow for Chiller 9.",             "forecasting"),
    ("T03", "Was there any abnormal behavior in Chiller 9 over the past week?",     "anomaly_detection"),
    ("T04", "What sensors are available for Chiller 6, and what do they measure?",  "metadata"),
    ("T05", "Chiller 9 vibration has been rising — should we schedule service?",    "fault_diagnosis"),
    ("T06", "Predict COP for Chiller 6 next month and flag if maintenance needed.", "forecasting"),
]


# ── Condition A: Baseline (direct LLM, no skills) ────────────────────────────

def run_condition_a(task: str) -> dict:
    t0 = time.time()
    answer = ""
    system = (
                "You are an industrial asset operations agent. "
                "Answer the following question about equipment maintenance."
            )
    try:
        if os.getenv("LLM_PROVIDER") == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": f"Task: {task}"}],
            )
            answer = resp.content[0].text.strip()
        elif os.getenv("LLM_PROVIDER") == "groq":
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=256,
                messages=[{"role": "system", "content": system},
                    {"role": "user", "content": f"Task: {task}"}]
            )
            answer = resp.choices[0].message.content.strip() 
    except Exception as e:
        answer = f"error: {e}"

    return {
        "task": task, "plan": ["direct_llm"], "result": {"answer": answer},
        "metrics": {"plan": ["direct_llm"], "tool_calls": 1,
                    "skills_skipped": [], "total_cost": 0.5,
                    "latency_s": round(time.time() - t0, 3)},
    }


# ── Condition B: +Planning only (no conditional exec, no knowledge) ──────────

def run_condition_b(task: str) -> dict:
    """Planner selects skills but executor skips no-skip checks and knowledge."""
    from agent import SkillAgent, _extract_asset
    from skills import SKILL_REGISTRY
    import time

    agent    = SkillAgent()
    asset_id = _extract_asset(task)
    plan     = agent.plan(task)
    context  = {}
    cost     = 0.0
    calls    = 0
    t0       = time.time()

    for skill_name in plan:
        skill = SKILL_REGISTRY.get(skill_name)
        if not skill:
            continue
        # No should_skip, no knowledge, no early stop
        try:
            result = skill["fn"](asset_id, context=context, task="")  # empty task = no knowledge
            context.update(result.get("output", {}))
            cost  += skill["cost"]
            calls += 1
        except Exception:
            pass

    return {
        "task": task, "plan": plan, "result": context,
        "metrics": {"plan": plan, "tool_calls": calls, "skills_skipped": [],
                    "total_cost": round(cost, 3), "latency_s": round(time.time() - t0, 3)},
    }


# ── Condition C: +Planning+Skills (no knowledge injection) ───────────────────

def run_condition_c(task: str) -> dict:
    """Planner + skills + conditional exec, but knowledge plugins disabled."""
    from agent import SkillAgent, _extract_asset
    from skills import SKILL_REGISTRY
    import time

    agent    = SkillAgent()
    asset_id = _extract_asset(task)
    plan     = agent.plan(task)
    context  = {}
    cost     = 0.0
    calls    = 0
    skipped  = []
    t0       = time.time()

    for skill_name in plan:
        skill = SKILL_REGISTRY.get(skill_name)
        if not skill:
            continue
        if skill["should_skip"](context):
            skipped.append(skill_name)
            continue
        try:
            # Pass empty task string → knowledge.get_knowledge() returns {} for all plugins
            result = skill["fn"](asset_id, context=context, task="")
            output = result.get("output", {})
            context.update(output)
            cost  += skill["cost"]
            calls += 1
            if result.get("should_stop"):
                break
        except Exception:
            pass

    return {
        "task": task, "plan": plan, "result": context,
        "metrics": {"plan": plan, "tool_calls": calls, "skills_skipped": skipped,
                    "total_cost": round(cost, 3), "latency_s": round(time.time() - t0, 3)},
    }


# ── Condition D: Full system ──────────────────────────────────────────────────

def run_condition_d(task: str) -> dict:
    from agent import SkillAgent
    return SkillAgent().run(task)


# ── Runner ────────────────────────────────────────────────────────────────────

CONDITIONS = {
    "A_baseline":           run_condition_a,
    "B_planning":           run_condition_b,
    "C_planning_skills":    run_condition_c,
    "D_full_system":        run_condition_d,
}

FIELDS = [
    "condition", "task_id", "category", "task", "plan",
    "tool_calls", "skipped_conditional", "skipped_early_stop", "skills_skipped",
    "total_cost", "latency_s", "task_completion", "error",
]


def evaluate_all(output_dir: str = "eval_results") -> None:
    Path(output_dir).mkdir(exist_ok=True)
    rows = []

    for cond_name, run_fn in CONDITIONS.items():
        print(f"\n{'─'*60}\nCondition: {cond_name}\n{'─'*60}")
        for task_id, task, category in TASK_BANK:
            print(f"  {task_id} [{category}] {task[:55]}...")
            try:
                out = run_fn(task)
                m   = out.get("metrics", {})
                rows.append({
                    "condition":           cond_name,
                    "task_id":             task_id,
                    "category":            category,
                    "task":                task[:80],
                    "plan":                json.dumps(m.get("plan", [])),
                    "tool_calls":          m.get("tool_calls", -1),
                    "skipped_conditional": len(m.get("skipped_conditional", [])),
                    "skipped_early_stop":  len(m.get("skipped_early_stop", [])),
                    "skills_skipped":      len(m.get("skills_skipped", [])),
                    "total_cost":          m.get("total_cost", -1),
                    "latency_s":           m.get("latency_s", -1),
                    "task_completion":     "",
                    "error":              "",
                })
                print(
                    f"    ✓  calls={m.get('tool_calls')} "
                    f"cond_skip={len(m.get('skipped_conditional',[]))} "
                    f"early_skip={len(m.get('skipped_early_stop',[]))} "
                    f"cost={m.get('total_cost')} lat={m.get('latency_s')}s"
                )
            except Exception as e:
                logger.error(f"    ✗  {task_id}: {e}")
                rows.append({
                    "condition": cond_name, "task_id": task_id, "category": category,
                    "task": task[:80], "plan": "", "tool_calls": -1, "skills_skipped": -1,
                    "total_cost": -1, "latency_s": -1, "task_completion": "", "error": str(e),
                })

    # Write CSV
    csv_path = Path(output_dir) / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅  Results → {csv_path}")
    _print_summary(rows)


def _print_summary(rows: list) -> None:
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in rows:
        if isinstance(r["tool_calls"], int) and r["tool_calls"] >= 0:
            by_cond[r["condition"]].append(r)

    print(f"\n{'Condition':<25} {'n':>4} {'avg_calls':>10} {'avg_cost':>10} {'avg_lat(s)':>12}")
    for cond, task_rows in sorted(by_cond.items()):
        n   = len(task_rows)
        avg = lambda k: sum(float(r[k]) for r in task_rows) / n
        print(f"{cond:<25} {n:>4} {avg('tool_calls'):>10.1f} {avg('total_cost'):>10.3f} {avg('latency_s'):>12.2f}")


if __name__ == "__main__":
    evaluate_all()
