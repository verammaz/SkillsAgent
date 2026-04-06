"""run.py — entry point for the AssetOps SkillAgent.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python run.py
"""
import json
# Vera:
from agent import SkillAgent

# Mana:
#from deep_agent import SkillAgent

# The four scenarios from the proposal
SCENARIOS = [
    "Why is Chiller 6 behaving abnormally and do we need a work order?",
    "Forecast next week's condenser water flow for Chiller 9 and determine if maintenance is needed.",
    "Was there any abnormal behavior in Chiller 9 over the past week?",
    "What sensors are available for Chiller 6, and what do they measure?",
]

if __name__ == "__main__":
    agent = SkillAgent()

    # Run a single query
    # query  = SCENARIOS[0]
    # result = agent.run(query)

    # print(f"\n{'='*65}")
    # print(f"Query : {query}")
    # print("=" * 65)
    # print(json.dumps(result["result"], indent=2, default=str))

    # m = result["metrics"]
    # print(f"\n── Metrics ──────────────────────────────────────")
    # print(f"  Plan:               {m['plan']}")
    # print(f"  Executed:           {m['skills_executed']}")
    # print(f"  Skipped (cond.):    {m['skipped_conditional']}")
    # print(f"  Skipped (no-reach): {m['skipped_early_stop']}")
    # print(f"  Stopped at:         {m['stopped_at']}")
    # print(f"  Tool calls:         {m['tool_calls']}")
    # print(f"  Total cost:         {m['total_cost']}")
    # print(f"  Latency:            {m['latency_s']}s")

    # Uncomment to sweep all four proposal scenarios:
    for query in SCENARIOS:
        print(f"\n{'='*65}")
        print(f"Query : {query}")
        print("=" * 65)
        result = agent.run(query)
        print(json.dumps(result["result"], indent=2, default=str))

        m = result["metrics"]
        print(f"\n── Metrics ──────────────────────────────────────")
        print(f"  Plan:               {m['plan']}")
        print(f"  Executed:           {m['skills_executed']}")
        print(f"  Skipped (cond.):    {m['skipped_conditional']}")
        print(f"  Skipped (no-reach): {m['skipped_early_stop']}")
        print(f"  Stopped at:         {m['stopped_at']}")
        print(f"  Tool calls:         {m['tool_calls']}")
        print(f"  Total cost:         {m['total_cost']}")
        print(f"  Latency:            {m['latency_s']}s")
