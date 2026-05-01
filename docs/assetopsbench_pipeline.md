# AssetOpsBench Scoring Pipeline

This document describes the recommended two-stage pipeline:

1. **Execute** agent conditions on a fixed scenario snapshot.
2. **Score** resulting outputs with the AssetOpsBench evaluator.

This keeps execution and grading decoupled and reproducible.

---

## Why two stages

- You can fetch scenarios once, reuse across reruns.
- You can rerun grading on subsets of conditions without recomputing agent outputs.
- `eval_runner.py` remains standalone and still supports built-in/HF/tsfm-report task banks.

---

## Stage 0: Start scenario server

Sync from the **workspace root** (`aobench`) so `scenario-client` / `scenario-server` resolve together:

```bash
cd AssetOpsBench/aobench
uv sync
uv run --directory scenario-server python serve.py
```

Server defaults to `http://localhost:8099`.

**If `pyarrow` fails to build with CMake (“Could not find ArrowConfig.cmake”)**: uv likely chose **Python 3.14+**, where `pyarrow` may have no wheel and tries to compile against Arrow C++. The aobench workspace is pinned to **Python 3.12–3.13** (`requires-python` and `aobench/.python-version`). Run `uv python install 3.12` if needed, delete `aobench/.venv`, then `uv sync` again so `pyarrow` installs from a wheel.

Useful scenario-set UUIDs:

- General: `d3bec9b0-59b4-4a2f-9497-28cb1eed1c80`
- IoT: `b3aa206a-f7dc-43c9-a1f4-dcf984417487`
- TSFM: `13aab653-66fe-4fe6-84d8-89f1b18eede3`
- Workorders: `4021467f-363b-41d2-8c62-f6aa738b01b7`

---

## Stage 1: Export scenario snapshot

```bash
cd SkillsAgent
python scripts/export_scenarios.py \
  --scenario-set 13aab653-66fe-4fe6-84d8-89f1b18eede3 \
  --output eval_inputs/tsfm_set/scenarios.jsonl
```

Output row schema (`jsonl`):

- `scenario_id`
- `query`
- `metadata`
- `scenario_set_id`
- `scenario_set_title`
- `fetched_at`

---

## Stage 2: Run ablations on snapshot

```bash
python -m eval_runner \
  --scenario-file eval_inputs/tsfm_set/scenarios.jsonl \
  --scenario-set-id 13aab653-66fe-4fe6-84d8-89f1b18eede3 \
  --conditions C D F E \
  --theta-values 0.5 0.6 0.65 0.7 0.8 0.9 0.95 \
  --output-dir eval_results/tsfm_snapshot_run \
  --trajectory-log eval_results/tsfm_snapshot_run/trajectories.jsonl
```

`ablation_results.csv` now includes:

- `scenario_set_id`
- `scenario_id`
- `result_json`
- `trace_json`

These are sufficient for deferred grading.

---

## Stage 3A: Score directly from eval CSV (no re-run)

```bash
python scripts/score_with_assetopsbench.py \
  --server-url http://localhost:8099 \
  --from-eval-csv eval_results/tsfm_snapshot_run/ablation_results.csv \
  --conditions C D F E \
  --output-dir eval_results/tsfm_snapshot_scored
```

Optional filters:

- `--theta-values 0.6 0.7 0.8`

---

## Stage 3B: Score in live-run mode (legacy)

This mode fetches scenarios and runs agent + grading in one script.

```bash
python scripts/score_with_assetopsbench.py \
  --scenario-set 13aab653-66fe-4fe6-84d8-89f1b18eede3 \
  --conditions C D F E \
  --output-dir eval_results/aob_tsfm
```

---

## Outputs

Scoring outputs:

- `grading_summary.csv` — condition-level correctness/accuracy
- `grading_details.csv` — scenario-level grade details (+ local metrics if live-run mode)
- `graded_<condition>.json` — raw AssetOpsBench grader response

Execution outputs:

- `ablation_results.csv`
- `trajectories.jsonl`

---

## Notes

- `run.py` is unchanged (quick local demo driver).
- `eval_runner.py` still works standalone without scenario snapshots.
- Scenario snapshot mode is additive and intended for reproducible benchmark runs.
