"""Tests for calibrated SKILL_COSTS loading + Condition~E cost budget."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


def _reload_skills_and_agent():
    """Re-import ``skills`` (which loads calibrated costs) and ``agent``."""
    for mod in ("agent", "skills"):
        if mod in sys.modules:
            del sys.modules[mod]
    import skills  # noqa: F401
    import agent   # noqa: F401
    return sys.modules["skills"], sys.modules["agent"]


def test_calibrated_costs_override_priors(tmp_path, monkeypatch):
    """``SKILL_COSTS_PATH`` must overwrite hand-set priors + drive DEEP_TSFM_COST."""
    path = tmp_path / "skill_costs.json"
    path.write_text(json.dumps({
        "data_retrieval":      42.0,
        "root_cause_analysis":  3.14,
        "__deep_tsfm__":        9.5,
    }))
    monkeypatch.setenv("SKILL_COSTS_PATH", str(path))
    monkeypatch.delenv("DEEP_TSFM_COST", raising=False)

    skills, agent = _reload_skills_and_agent()

    assert skills.SKILL_REGISTRY["data_retrieval"]["cost"] == 42.0
    assert skills.SKILL_REGISTRY["root_cause_analysis"]["cost"] == pytest.approx(3.14)
    # unspecified keys keep their prior
    assert skills.SKILL_REGISTRY["validate_failure"]["cost"] == 0.3
    assert skills.CALIBRATED_COSTS.get("__deep_tsfm__") == 9.5
    assert agent.DEEP_TSFM_COST == 9.5


def test_explicit_deep_tsfm_env_beats_calibrated(tmp_path, monkeypatch):
    """``DEEP_TSFM_COST`` env var wins over the JSON's __deep_tsfm__."""
    path = tmp_path / "skill_costs.json"
    path.write_text(json.dumps({"__deep_tsfm__": 9.5}))
    monkeypatch.setenv("SKILL_COSTS_PATH", str(path))
    monkeypatch.setenv("DEEP_TSFM_COST", "2.0")

    _, agent = _reload_skills_and_agent()
    assert agent.DEEP_TSFM_COST == 2.0


def test_condition_e_cost_budget_default(monkeypatch, tmp_path):
    """Without ``COST_BUDGET`` env, budget is 80% of full-plan cost."""
    monkeypatch.delenv("COST_BUDGET", raising=False)
    monkeypatch.setenv("SKILL_COSTS_PATH", str(tmp_path / "missing.json"))
    monkeypatch.delenv("DEEP_TSFM_COST", raising=False)

    for mod in ("eval_runner", "agent", "skills"):
        sys.modules.pop(mod, None)

    from eval_runner import _cost_budget_for_condition_e
    from skills import SKILL_REGISTRY
    from agent import DEEP_TSFM_COST

    expected = round((sum(m["cost"] for m in SKILL_REGISTRY.values()) + DEEP_TSFM_COST) * 0.8, 3)
    assert _cost_budget_for_condition_e() == expected


def test_condition_e_cost_budget_env_override(monkeypatch):
    for mod in ("eval_runner", "agent", "skills"):
        sys.modules.pop(mod, None)

    monkeypatch.setenv("COST_BUDGET", "1.25")
    from eval_runner import _cost_budget_for_condition_e
    assert _cost_budget_for_condition_e() == 1.25

    monkeypatch.setenv("COST_BUDGET", "none")
    sys.modules.pop("eval_runner", None)
    from eval_runner import _cost_budget_for_condition_e as fn2
    assert fn2() is None


def test_agent_skips_skills_over_budget(monkeypatch, tmp_path):
    """Setting a tiny budget causes SkillAgent to skip the expensive tail."""
    # Force skills.py to use the hardcoded priors, not any calibrated JSON on disk.
    monkeypatch.setenv("SKILL_COSTS_PATH", str(tmp_path / "missing.json"))
    monkeypatch.delenv("DEEP_TSFM_COST", raising=False)

    for mod in ("agent", "skills"):
        sys.modules.pop(mod, None)

    from agent import SkillAgent

    agent = SkillAgent(cost_budget=0.3)
    # Stub the planner so this test doesn't need an LLM
    agent.plan = lambda task: [
        "data_retrieval",        # cost 0.2  -> runs, total=0.2
        "root_cause_analysis",   # cost 0.8  -> skipped (would push total to 1.0)
        "work_order_generation", # cost 0.2  -> runs, total=0.4? no, 0.2+0.2=0.4 > 0.3 -> skipped
    ]

    result = agent.run("Diagnose anomalies on Chiller 6")
    m = result["metrics"]
    assert m["total_cost"] <= 0.3 + 1e-6, m
    # Given costs 0.2 + 0.8 + 0.2 against budget 0.3, RCA (0.8) is guaranteed
    # skipped (over budget). data_retrieval (0.2) runs first and uses up the budget.
    assert "root_cause_analysis" in m["skipped_conditional"], m
    assert "data_retrieval" in m["skills_executed"], m
