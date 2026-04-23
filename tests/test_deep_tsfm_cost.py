"""DEEP_TSFM_COST is charged exactly when the Confidence Evaluator fires."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import skills  # noqa: E402  # bound at import time; re-imported per test below


def _stub_rca_factory(*, deep_invoked: bool, cost_bump_expected: float):
    def _stub_rca(asset_id, context, task):
        return {
            "output": {
                "failure": "compressor_overheating",
                "severity": "high",
                "diagnosis_confidence": 0.84 if deep_invoked else 0.92,
                "deep_tsfm_invoked": deep_invoked,
                "anomalies_detected": True,
            },
            "should_stop": False,
        }
    return _stub_rca


def _install_minimal_plan(monkeypatch, *, deep_invoked: bool):
    """Force plan = [root_cause_analysis] and stub its body.

    Pin the ``cost`` back to the 0.8 prior so a calibrated ``skill_costs.json``
    on disk doesn't perturb the arithmetic these tests were written against.
    Reload ``skills`` first so this test is robust against earlier tests that
    may have popped the module and loaded alternate ``skill_costs.json`` files.
    """
    import sys as _sys
    _sys.modules.pop("agent", None)
    _sys.modules.pop("skills", None)
    import skills as _skills  # freshly loaded — bound to current SKILL_COSTS_PATH

    monkeypatch.setitem(
        _skills.SKILL_REGISTRY,
        "root_cause_analysis",
        {
            **_skills.SKILL_REGISTRY["root_cause_analysis"],
            "fn": _stub_rca_factory(
                deep_invoked=deep_invoked, cost_bump_expected=1.0
            ),
            "should_skip": lambda _ctx: False,
            "cost": 0.8,
        },
    )

    import agent as agent_mod
    monkeypatch.setattr(
        agent_mod.SkillAgent, "plan", lambda self, task: ["root_cause_analysis"]
    )
    return agent_mod


def test_deep_tsfm_cost_added_when_invoked(monkeypatch):
    monkeypatch.setenv("DEEP_TSFM_COST", "1.0")
    agent_mod = _install_minimal_plan(monkeypatch, deep_invoked=True)
    result = agent_mod.SkillAgent().run("Why is Chiller 6 abnormal?")
    # RCA base cost is 0.8 + DEEP_TSFM_COST 1.0 = 1.8
    assert result["metrics"]["total_cost"] == 1.8
    assert result["metrics"]["deep_tsfm_invoked"] is True


def test_deep_tsfm_cost_not_added_when_skipped(monkeypatch):
    monkeypatch.setenv("DEEP_TSFM_COST", "1.0")
    agent_mod = _install_minimal_plan(monkeypatch, deep_invoked=False)
    result = agent_mod.SkillAgent().run("Why is Chiller 6 abnormal?")
    assert result["metrics"]["total_cost"] == 0.8
    assert result["metrics"]["deep_tsfm_invoked"] is False


def test_deep_tsfm_cost_is_env_tunable(monkeypatch):
    monkeypatch.setenv("DEEP_TSFM_COST", "2.5")
    agent_mod = _install_minimal_plan(monkeypatch, deep_invoked=True)
    result = agent_mod.SkillAgent().run("Why is Chiller 6 abnormal?")
    assert result["metrics"]["total_cost"] == 0.8 + 2.5
