"""SkillAgent: LLM planner + knowledge-aware executor for AssetOpsBench tasks.

Layers:
    1. Planner — LLM produces an ordered skill list; heuristic fallback on failure.
    2. Confidence Evaluator (`confidence_evaluator.should_invoke_deep_tsfm`) —
       gates `deep_tsfm_refine_anomalies` inside `root_cause_analysis` when the
       FMSR-proxy confidence is below `RCA_CONFIDENCE_THETA` (proposal cond. E).
    3. Executor — runs each skill with (a) `should_skip(context)`,
       (b) `should_stop` early termination, (c) `cost_budget` skipping,
       (d) `get_knowledge()` injection.
"""

import dotenv
dotenv.load_dotenv()

import os
import re
import json
import logging
import time
from typing import Optional

from skills import CALIBRATED_COSTS, SKILL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger(__name__)

# Cost charged on top of RCA when the Confidence Evaluator invokes deep TSFM.
# Order: DEEP_TSFM_COST env var > calibrated `__deep_tsfm__` > 1.0.
_env_deep = os.getenv("DEEP_TSFM_COST")
if _env_deep is not None:
    DEEP_TSFM_COST = float(_env_deep)
elif isinstance(CALIBRATED_COSTS.get("__deep_tsfm__"), (int, float)):
    DEEP_TSFM_COST = float(CALIBRATED_COSTS["__deep_tsfm__"])
else:
    DEEP_TSFM_COST = 1.0

# ── Planner prompt ────────────────────────────────────────────────────────────

_SKILL_LIST = "\n".join(
    f"  {name} (cost={meta['cost']}): {meta['description']}"
    for name, meta in SKILL_REGISTRY.items()
)

PLANNER_SYSTEM = f"""\
You are a planning agent for industrial asset operations and maintenance.

Available skills:
{_SKILL_LIST}

Select the minimal ordered skill sequence to complete the task.

Ordering rules:
  - data_retrieval must come before anomaly_detection, root_cause_analysis, forecasting.
  - validate_failure must come after root_cause_analysis.
  - work_order_generation is terminal; include only if a fault is likely.
  - For metadata-only queries, return ["metadata_retrieval"].
  - For forecasting queries, return ["data_retrieval", "forecasting"].
  - For anomaly queries: ["data_retrieval", "anomaly_detection", "root_cause_analysis"].
  - For fault + work order: add "validate_failure", "work_order_generation" after RCA.

Return ONLY a valid JSON array of skill names. No explanation, no markdown.

For fault-diagnosis tasks, root_cause_analysis internally runs lightweight FMSR
mapping then may invoke deep TSFM refinement when diagnosis confidence is below
RCA_CONFIDENCE_THETA (env, default 0.8); do not add a separate skill for that.
"""


class SkillAgent:
    """Skill-augmented, knowledge-aware agent for AssetOpsBench tasks.

    Usage:
        agent = SkillAgent()
        result = agent.run("Why is Chiller 6 behaving abnormally?")
        print(result)
    """

    def __init__(self, cost_budget: Optional[float] = None):
        self.cost_budget = cost_budget
        self.provider: Optional[str] = None

    # ── Planner ───────────────────────────────────────────────────────────────

    def plan(self, task: str) -> list:
        """Return an ordered skill plan (LLM first, heuristic fallback)."""
        try:
            from skills import _call_llm
            raw = _call_llm(PLANNER_SYSTEM, f"Task: {task}", max_tokens=256).strip()
            self.provider = (os.getenv("LLM_PROVIDER") or "watsonx").lower()
            if not raw:
                raise RuntimeError("empty response from all LLM providers")
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            plan = json.loads(match.group(0) if match else raw)
            if not isinstance(plan, list):
                raise ValueError(f"Non-list plan: {plan}")
            plan = [s for s in plan if s in SKILL_REGISTRY]
            logger.info(f"Planner (LLM) → {plan}")
            return plan
        except Exception as e:
            logger.warning(f"Planner LLM failed ({e}), using heuristic.")
            return self._heuristic_plan(task)

    def _heuristic_plan(self, task: str) -> list:
        t = task.lower()
        if any(w in t for w in ("what sensor", "describe sensor", "metadata", "available")):
            return ["metadata_retrieval"]
        if any(w in t for w in ("forecast", "predict", "next week", "future")):
            return ["data_retrieval", "forecasting", "validate_failure", "work_order_generation"]
        if any(w in t for w in ("anomal", "abnormal", "unusual", "behav")):
            return ["data_retrieval", "anomaly_detection", "root_cause_analysis",
                    "validate_failure", "work_order_generation"]
        if any(w in t for w in ("why", "fault", "failure", "work order", "maintenance")):
            return ["data_retrieval", "anomaly_detection", "root_cause_analysis",
                    "validate_failure", "work_order_generation"]
        return ["data_retrieval", "anomaly_detection", "root_cause_analysis"]

    # ── Executor ──────────────────────────────────────────────────────────────

    def run(self, task: str) -> dict:
        """Plan then execute with cost-aware conditional logic."""
        asset_id   = _extract_asset(task)
        plan       = self.plan(task)
        logger.info(f"Task: {task[:80]}")
        logger.info(f"Asset: {asset_id}  |  Plan: {plan}")
 
        context             = {}
        total_cost          = 0.0
        tool_calls          = 0
        skipped_conditional = []   # should_skip() fired or over budget
        executed            = []   # skills that actually ran
        stopped_at          = None # skill that triggered early stop
        skill_steps         = []   # per-skill trajectory (for JSONL logs)
        t0                  = time.time()

        n_skills = len(plan)
        for i, skill_name in enumerate(plan):
            skill = SKILL_REGISTRY.get(skill_name)
            if not skill:
                logger.warning(f"Unknown skill '{skill_name}', skipping.")
                skipped_conditional.append(skill_name)
                continue

            if self.cost_budget and (total_cost + skill["cost"]) > self.cost_budget:
                logger.info(f"  ⊘ '{skill_name}': over budget (cost={skill['cost']}).")
                skipped_conditional.append(skill_name)
                continue

            if skill["should_skip"](context):
                logger.info(f"  ⊘ '{skill_name}': condition not met, skipping.")
                skipped_conditional.append(skill_name)
                continue

            logger.info(f"  ▶ {skill_name}")
            try:
                result = skill["fn"](asset_id, context=context, task=task)
            except Exception as e:
                logger.error(f"  ✗ '{skill_name}' raised: {e}", exc_info=True)
                continue

            out = result.get("output", {})
            context.update(out)
            total_cost += skill["cost"]
            # Deep TSFM fires inside root_cause_analysis. Charge it here so
            # Condition D (deep off) and Condition E (deep gated) are
            # cost-comparable per plan.
            if out.get("deep_tsfm_invoked"):
                total_cost += DEEP_TSFM_COST
            tool_calls += 1
            executed.append(skill_name)
            skill_steps.append({
                "skill": skill_name,
                "output_keys": sorted(out.keys()),
                "should_stop": bool(result.get("should_stop", False)),
            })

            should_stop = result.get("should_stop", False)
            logger.info(f"    should_stop={should_stop}")
            if i != n_skills - 1 and should_stop:
                logger.info(f"  ⏹  Early stop: '{skill_name}' signalled termination.")
                stopped_at = skill_name
                break


        executed_set       = set(executed)
        conditional_set    = set(skipped_conditional)
        skipped_early_stop = [
            s for s in plan
            if s not in executed_set and s not in conditional_set
        ]
 
        all_skipped = skipped_conditional + skipped_early_stop
        if skipped_early_stop:
            logger.info(f"  ⊘ Not reached (early stop): {skipped_early_stop}")
 
        metrics = {
            "plan":                  plan,
            "tool_calls":            tool_calls,
            "skills_executed":       executed,
            "skills_skipped":        all_skipped,
            "skipped_conditional":   skipped_conditional,
            "skipped_early_stop":    skipped_early_stop,
            "stopped_at":            stopped_at,
            "skill_steps":           skill_steps,
            "total_cost":            round(total_cost, 3),
            "latency_s":             round(time.time() - t0, 3),
            "diagnosis_confidence":          context.get("diagnosis_confidence"),
            "diagnosis_confidence_pre_deep": context.get("diagnosis_confidence_pre_deep"),
            "deep_tsfm_invoked":             bool(context.get("deep_tsfm_invoked", False)),
        }

        payload = {
            "task":    task,
            "asset":   asset_id,
            "result":  context,
            "metrics": metrics,
        }
        if os.getenv("TRAJECTORY_LOG_PATH"):
            from trajectory_log import append_trajectory_line, build_agent_trajectory

            append_trajectory_line(
                build_agent_trajectory(
                    task=task,
                    asset_id=asset_id,
                    plan=plan,
                    metrics=metrics,
                    context=context,
                    skill_steps=skill_steps,
                )
            )
        return payload
 

def _extract_asset(task: str) -> str:
    m = re.search(r"chiller\s*(\d+)", task.lower())
    return f"Chiller {m.group(1)}" if m else "Chiller 6"
