"""Measure per-skill median wall-clock latency and write ``skill_costs.json``.

Run once per deployment (local Mac, Colab T4, GCP VM, ...) before the ablation
so Condition E's ``cost_budget`` reflects real wall-clock — not hand-assigned
priors. The resulting JSON is loaded by ``skills._load_calibrated_costs()``.

Usage::

    python scripts/calibrate_costs.py --runs 3 \
        --task "Diagnose anomalies on Chiller 6 and generate a work order" \
        --output skill_costs.json

Each skill is invoked in isolation (no planner) with a pre-built context so we
time only the skill body, not the LLM planner.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Let ``python scripts/calibrate_costs.py`` find the top-level modules.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
os.chdir(_PKG_ROOT)

from dotenv import load_dotenv

from skills import SKILL_REGISTRY
from tools import (
    deep_tsfm_refine_anomalies,
    detect_anomaly,
    get_sensor_data,
)


DEFAULT_TASK = "Diagnose anomalies on Chiller 6 and generate a work order"


DEFAULT_ASSET = "Chiller 6"


def _build_warm_context(task: str) -> dict:
    """Prefetch IoT data once so skill runs don't all pay the subprocess cost."""
    data = get_sensor_data(DEFAULT_ASSET, lookback_days=14)
    basic = detect_anomaly(data)
    return {
        "task": task,
        "sensor_data": data,
        "anomaly": basic,
        "failure_modes": None,
    }


def _time_skill(name: str, task: str, context: dict, runs: int) -> tuple[float, list[float]]:
    meta = SKILL_REGISTRY[name]
    fn = meta["fn"]
    times: list[float] = []
    for _ in range(runs):
        ctx = dict(context)  # fresh copy so prior skill's output doesn't short-circuit
        t0 = time.perf_counter()
        try:
            out = fn(DEFAULT_ASSET, context=ctx, task=task)
        except Exception as exc:
            print(f"  !! {name} raised {type(exc).__name__}: {exc}")
            out = None
        dt = time.perf_counter() - t0
        times.append(dt)
        if out is not None and isinstance(out, dict):
            context.update(out)
    return statistics.median(times), times


def _time_deep_tsfm(context: dict, runs: int) -> float:
    data = context.get("sensor_data") or get_sensor_data(DEFAULT_ASSET, lookback_days=14)
    basic = context.get("anomaly") or detect_anomaly(data)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            deep_tsfm_refine_anomalies(DEFAULT_ASSET, data, basic)
        except Exception as exc:
            print(f"  !! deep_tsfm raised {type(exc).__name__}: {exc}")
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def calibrate(*, task: str, runs: int) -> dict:
    ctx = _build_warm_context(task)
    out: dict = {}

    print(f"Calibrating {len(SKILL_REGISTRY)} skills × {runs} runs (task: {task!r})\n")
    for name in SKILL_REGISTRY:
        median, raw = _time_skill(name, task, ctx, runs=runs)
        out[name] = round(median, 4)
        print(f"  {name:<24s} median={median:7.3f}s   runs={['%.3f' % t for t in raw]}")

    deep_median = _time_deep_tsfm(ctx, runs=runs)
    out["__deep_tsfm__"] = round(deep_median, 4)
    print(f"\n  __deep_tsfm__            median={deep_median:7.3f}s")

    return out


def main() -> None:
    load_dotenv(".env")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=3, help="Repetitions per skill (default 3).")
    ap.add_argument("--task", default=DEFAULT_TASK, help="Sample task fed to each skill.")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("skill_costs.json"),
        help="Output JSON with median latencies (seconds) per skill.",
    )
    ns = ap.parse_args()

    result = calibrate(task=ns.task, runs=ns.runs)
    ns.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {ns.output}  ({sum(v for k, v in result.items() if not k.startswith('__')):.2f}s total per full plan)")


if __name__ == "__main__":
    main()
