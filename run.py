"""Quick benchmark driver: run SkillAgent on four representative scenarios.

Usage:
    python run.py
"""
import json

from agent import SkillAgent

SCENARIOS = [
    "Why is Chiller 6 behaving abnormally and do we need a work order?",
    "Forecast next week's condenser water flow for Chiller 6 and determine if maintenance is needed.",
    "Was there any abnormal behavior in Chiller 6 over the past week?",
    "What sensors are available for Chiller 6, and what do they measure?",
]


def _print_metrics(m: dict) -> None:
    print(f"\n── Metrics ──────────────────────────────────────")
    print(f"  Plan:               {m['plan']}")
    print(f"  Executed:           {m['skills_executed']}")
    print(f"  Skipped (cond.):    {m['skipped_conditional']}")
    print(f"  Skipped (no-reach): {m['skipped_early_stop']}")
    print(f"  Stopped at:         {m['stopped_at']}")
    print(f"  Tool calls:         {m['tool_calls']}")
    print(f"  Total cost:         {m['total_cost']}")
    print(f"  Latency:            {m['latency_s']}s")
    print(f"  Deep TSFM invoked:  {m['deep_tsfm_invoked']}")
    print(f"  Confidence (pre):   {m['diagnosis_confidence_pre_deep']}")
    print(f"  Confidence (final): {m['diagnosis_confidence']}")


if __name__ == "__main__":
    agent = SkillAgent()
    for query in SCENARIOS:
        print(f"\n{'=' * 65}")
        print(f"Query : {query}")
        print("=" * 65)
        out = agent.run(query)
        print(json.dumps(out["result"], indent=2, default=str))
        _print_metrics(out["metrics"])
