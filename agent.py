"""agent.py

SkillAgent: the unified entry point for AssetOpsBench skill-augmented reasoning.

Architecture:
    1. Planner   — LLM generates an ordered skill plan; heuristic fallback if unavailable.
    2. Executor  — runs each skill with four efficiency optimizations:
                     (a) conditional execution  via skill.should_skip(context)
                     (b) early stopping         via should_stop flag or confidence threshold
                     (c) cost-aware skipping    if a cost budget is set
                     (d) knowledge injection    via get_knowledge() per skill
    3. Skills    — atomic, reusable workflows in skills.py
    4. Knowledge — targeted domain knowledge injected per skill in knowledge.py
    5. Tools     — AssetOpsBench agent wrappers in tools.py
"""

import dotenv
dotenv.load_dotenv()  # load .env file if present

import os
import re
import json
import logging
import time
from typing import Optional

from skills import SKILL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.90   # early stop when confidence exceeds this

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
        self.provider = None  # placeholder for LLM provider (e.g., Anthropic client)
        self._client = None   # lazy-init Anthropic client

    # ── Planner ───────────────────────────────────────────────────────────────

    def plan(self, task: str) -> list:
        """Generate an ordered skill plan. Falls back to heuristic."""
        try:
            client = self._get_client()
            raw = ""
            if self.provider == "anthropic":
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=256,
                    system=PLANNER_SYSTEM,
                    messages=[{"role": "user", "content": f"Task: {task}"}],
                )
                raw = resp.content[0].text.strip()
            elif self.provider == "groq":
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=256,
                    messages=[{"role": "system", "content": PLANNER_SYSTEM},
                        {"role": "user", "content": f"Task: {task}"}]
                )
                raw = resp.choices[0].message.content.strip() 
            
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            plan  = json.loads(match.group(0) if match else raw)
            # TODO: add check of plan?
            if not isinstance(plan, list):
                raise ValueError(f"Non-list plan: {plan}")
            # Validate — only keep known skills
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
        t0                  = time.time()

        n_skills = len(plan)
        for (i, skill_name) in enumerate(plan):
            skill = SKILL_REGISTRY.get(skill_name)
            if not skill:
                logger.warning(f"Unknown skill '{skill_name}', skipping.")
                skipped_conditional.append(skill_name)
                continue
 
            # (c) Cost-aware: skip if over budget
            if self.cost_budget and (total_cost + skill["cost"]) > self.cost_budget:
                logger.info(f"  ⊘ '{skill_name}': over budget (cost={skill['cost']}).")
                skipped_conditional.append(skill_name)
                continue
 
            # (a) Conditional execution
            if skill["should_skip"](context):
                logger.info(f"  ⊘ '{skill_name}': condition not met, skipping.")
                skipped_conditional.append(skill_name)
                continue
 
            # Execute
            logger.info(f"  ▶ {skill_name}")
            try:
                result = skill["fn"](asset_id, context=context, task=task)
            except Exception as e:
                logger.error(f"  ✗ '{skill_name}' raised: {e}", exc_info=True)
                continue
 
            context.update(result.get("output", {}))
            total_cost += skill["cost"]
            tool_calls += 1
            executed.append(skill_name)
 
            # confidence  = result.get("confidence", 0.5) TODO: add confidence to skill outputs?
            should_stop = result.get("should_stop", False)
            #logger.info(f"    confidence={confidence}")
            logger.info(f"    should_stop={should_stop}")
 
            # (b) Early stopping
            if i != n_skills - 1 and should_stop:
                logger.info(f"  ⏹  Early stop: '{skill_name}' signalled termination.")
                stopped_at = skill_name
                break
            # TODO: add 'confidence' based stopping as well 
 
        # Skills that were planned but never reached due to early stop
        executed_set       = set(executed)
        conditional_set    = set(skipped_conditional)
        skipped_early_stop = [
            s for s in plan
            if s not in executed_set and s not in conditional_set
        ]
 
        all_skipped = skipped_conditional + skipped_early_stop
        if skipped_early_stop:
            logger.info(f"  ⊘ Not reached (early stop): {skipped_early_stop}")
 
        return {
            "task":    task,
            "asset":   asset_id,
            "result":  context,
            "metrics": {
                "plan":                  plan,
                "tool_calls":            tool_calls,
                "skills_executed":       executed,
                "skills_skipped":        all_skipped,
                "skipped_conditional":   skipped_conditional,
                "skipped_early_stop":    skipped_early_stop,
                "stopped_at":            stopped_at,
                "total_cost":            round(total_cost, 3),
                "latency_s":             round(time.time() - t0, 3),
            },
        }
 

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            if os.getenv("LLM_PROVIDER") == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                self.provider = "anthropic"
            elif os.getenv("LLM_PROVIDER") == "groq":
                from groq import Groq
                self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                self.provider = "groq"
        
        return self._client


def _extract_asset(task: str) -> str:
    m = re.search(r"chiller\s*(\d+)", task.lower())
    return f"Chiller {m.group(1)}" if m else "Chiller 6"
