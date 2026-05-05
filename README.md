# SkillsAgent

Skill-augmented, knowledge-aware agent for AssetOpsBench fault-diagnosis tasks.
The agent plans a sequence of **skills**, injects targeted **domain knowledge**,
calls real AssetOpsBench MCP servers (IoT / FMSR / TSFM / WO) via `uv run`
subprocesses, and invokes **deep TSFM refinement** when initial diagnosis
confidence falls below a threshold (proposal Alg. 2).

```
            ┌──────────────────── SkillAgent.run(task) ─────────────────────--┐
            │                                                                 │
task  ───►  │  1. plan(task)       (LLM: watsonx / gemini / anthropic / groq) │
            │  2. for each skill:                                             │
            │       a. cost-budget gate   → skipped_conditional               │
            │       b. should_skip(ctx)   → skipped_conditional               │
            │       c. get_knowledge()    (6 plugins, scoped per skill)       │
            │       d. call tool(s)       (IoT / FMSR / TSFM / WO via uv)     │
            │       e. optional LLM enrichment                                │
            │       f. if RCA conf < θ    → deep_tsfm_refine_anomalies()      │
            │  3. early-stop if should_stop                                   │
            │                                                                 │
            └──────► metrics: plan, skills_executed, skills_skipped,          │
                     total_cost, latency_s, deep_tsfm_invoked,                │
                     diagnosis_confidence_pre_deep, diagnosis_confidence,     │
                     task_completion                                          │
```

---

## Table of contents
1. [Architecture](#architecture)
2. [Seven Skills](#seven-skills)
3. [Six Knowledge plugins](#six-knowledge-plugins)
4. [Conditional deep TSFM (proposal Alg. 2)](#conditional-deep-tsfm)
5. [Graded diagnosis confidence](#graded-diagnosis-confidence)
6. [Cost-aware execution](#cost-aware-execution)
7. [LLM provider routing](#llm-provider-routing)
8. [Data sources (real vs. mock)](#data-sources)
9. [Repo layout](#repo-layout)
10. [Local setup](#local-setup)
11. [Colab setup (T4 GPU)](#colab-setup)
12. [Running the agent](#running-the-agent)
13. [Running the ablation](#running-the-ablation)
14. [Calibrating skill costs](#calibrating-skill-costs)
15. [Evaluation conditions](#evaluation-conditions)
16. [Results](#results)
17. [Environment variables](#environment-variables)
18. [Tests](#tests)

---

## Architecture

`SkillAgent` (`agent.py`) is a three-layer planner/executor:

| Layer | File | Role |
|---|---|---|
| Planner | `agent.py::plan` | LLM prompt → ordered list of skill names (heuristic fallback) |
| Confidence Evaluator | `confidence_evaluator.py` | Gates deep-TSFM invocation on `confidence < θ` |
| Executor | `agent.py::run` | Runs skills; applies `should_skip`, `should_stop`, `cost_budget`; tracks `total_cost` |

Skills call functions in `tools.py`, which **primarily** talk to real
AssetOpsBench MCP servers and fall back to mock only when the server or its
data is unavailable.

---

## Seven Skills

Each skill is a function `fn(asset_id, context, task) -> {output, should_stop}`,
registered in `SKILL_REGISTRY` at the bottom of `skills.py` with `fn`,
`should_skip(context)`, `cost`, and `description`. Adding a skill is purely
additive — the executor reads only this dict.

| # | Skill | Real tool | Knowledge consumed | `should_skip` | `should_stop` |
|---|---|---|---|---|---|
| 1 | `data_retrieval` | `get_sensor_data` → IoT MCP (or `IOT_CSV_DIR` CSVs) | `time_series_metadata.default_lookback_days` | never | never |
| 2 | `metadata_retrieval` | `get_asset_metadata` → IoT MCP + `fetch_tsfm_catalog` static lookup | `sensor_metadata` + optional TSFM task/model catalog | never | always (self-contained) |
| 3 | `anomaly_detection` | local profile + IQR (`detect_anomaly`) | `sensor_thresholds`, `operating_ranges` | never | `severity == "none"` |
| 4 | `root_cause_analysis` | `map_failure_with_meta` → FMSR MCP; conditionally `deep_tsfm_refine_anomalies` → TSFM MCP | `failure_modes`, `anomaly_definition` | no anomalies detected | never |
| 5 | `validate_failure` | local policy check | `maintenance_policy` | no failure diagnosed OR no anomalies | no work order needed |
| 6 | `forecasting` | `forecast_sensor` → TSFM MCP | `operating_ranges`, `time_series_metadata`, `sensor_metadata` | never | forecast within operating range |
| 7 | `work_order_generation` | `generate_work_order` → WO MCP (reads local CSVs) | `maintenance_policy.auto_escalate` | `work_order_needed == False` | always (terminal) |

Every skill shares the same three-step pattern:

1. `get_knowledge(skill_name, task, context)` pulls relevant plugin outputs
2. Call one or more `tools.py` functions (real AssetOpsBench via `uv run`)
3. Optional LLM enrichment via `_call_llm(...)` (provider chain below)

---

## Six Knowledge plugins

Plugins live in `knowledge.py`. Each implements:

```python
name: str
relevant_skills: set[str]          # empty = all
retrieve(skill_name, task, context) -> dict
```

`get_knowledge(skill_name, ...)` merges the dicts returned by every plugin
whose `relevant_skills` matches the current skill. Setting
`KNOWLEDGE_INJECTION=0` disables the whole layer (Condition C).

| Plugin | `relevant_skills` | Injects | Source |
|---|---|---|---|
| `SensorMetadataPlugin` | `data_retrieval`, `anomaly_detection`, `metadata_retrieval` | `{asset_id, sensor_metadata}` | Static catalog **+ live merge from `main.json`** via `couch_export_catalog` when `COUCHDB_EXPORT_PATH` is set |
| `FailureModePlugin` | `root_cause_analysis`, `anomaly_detection`, `validate_failure` | `{failure_modes: [...]}` | Static: 6 canonical chiller failure modes with symptoms + severity |
| `MaintenancePolicyPlugin` | `validate_failure`, `work_order_generation` | `{maintenance_policy}` — `requires_work_order`, `response_times`, `auto_escalate` | Static policy derived from proposal |
| `OperatingRangesPlugin` | `anomaly_detection`, `forecasting`, `root_cause_analysis` | `{operating_ranges, sensor_thresholds}` | Static per-asset min/max per sensor |
| `AnomalyDefinitionPlugin` | `root_cause_analysis` | `{anomaly_definition}` — `min_context_rows`, semantics | Static (parameterises deep TSFM call) |
| `TimeSeriesMetadataPlugin` | `forecasting`, `data_retrieval` | `{time_series_metadata}` — resolution, lookback, seasonality, `lag_features` | Static |

`SensorMetadataPlugin` is the one plugin that ingests the provided `main.json`
— it streams the CouchDB export and merges discovered sensor columns into the
static catalog. The other five are intentionally small static proposal-derived
domain constants, meant to be *prompt-injected* rather than exhaustive
databases.

---

## Conditional deep TSFM

Proposal Alg. 2: FMSR diagnoses first; if `confidence < θ` the Confidence
Evaluator fires the deep TSAD subprocess (`run_integrated_tsad` on the primary
series), re-diagnoses with the refined anomaly, and records
`deep_tsfm_invoked=True` in the output.

The gate lives inside `root_cause_analysis` (`skills.py`):

```python
theta = theta_from_env()                                        # RCA_CONFIDENCE_THETA
if should_invoke_deep_tsfm(confidence_pre, theta=theta):
    anomaly = deep_tsfm_refine_anomalies(asset_id, sensor_data, anomaly,
                                         anomaly_definition=anomaly_def)
    meta = map_failure_with_meta(anomaly, failure_modes, asset_id=asset_id)
    failure = meta["failure"]
    confidence = score_diagnosis_confidence(anomaly, meta, task=task, ...)
    deep_invoked = True
```

The executor charges `DEEP_TSFM_COST` on top of RCA's own cost whenever
`deep_tsfm_invoked` is true, so Condition D (no deep TSFM) and Condition E
(deep TSFM enabled) are cost-comparable.

---

## Graded diagnosis confidence

`tools.score_diagnosis_confidence` combines six signals so the θ sweep
produces a continuous cost/accuracy curve rather than a step at one bucket
edge. Weights sum to 1.0 and output is clipped to `[0.05, 0.98]`.

| # | Signal | Weight | What it measures |
|---|---|---|---|
| 1 | `via_score` | 0.30 | Match quality: `fmsr` (0.90) > `knowledge` (0.60) > unknown (0.15); heavy penalty if `failure == "unknown_failure"` |
| 2 | `sev_score` | 0.22 | Severity bucket + log-scaled anomaly-detail density |
| 3 | `coverage_score` | 0.18 | Fraction of monitored sensors flagged, peaked at ~30% (too few → spurious; too many → systemic) |
| 4 | `tsad_score` | 0.13 | Non-zero integrated-TSAD conformal records (populated after deep TSFM; saturates at 20) |
| 5 | `task_specificity_score` | 0.12 | Does the prompt name a subsystem consistent with the diagnosis? (0.3 vague → 1.0 exact match) |
| 6 | `wo_score` | 0.05 | Past work order for this asset mentions the diagnosed failure |

Signals 3–6 degrade gracefully when inputs are missing, so the legacy two-arg
call `score_diagnosis_confidence(anomaly, meta)` still works.

---

## Cost-aware execution

Each skill carries a `cost` field. The executor accumulates `total_cost` and
skips any skill that would push the running total past `cost_budget`:

```python
if self.cost_budget and (total_cost + skill["cost"]) > self.cost_budget:
    skipped_conditional.append(skill_name)
    continue
```

Costs are **calibrated** from real wall-clock measurements via
`scripts/calibrate_costs.py`, which runs each skill N times on a warm context
and writes `skill_costs.json`. At import time `skills._load_calibrated_costs()`
picks this file up (path: `SKILL_COSTS_PATH` env or `./skill_costs.json`) and
overrides the hand-set priors. `DEEP_TSFM_COST` honours the same file's
`__deep_tsfm__` key unless overridden by the `DEEP_TSFM_COST` env var.

Condition E's cost budget is set in `eval_runner.py::_cost_budget_for_condition_e`:

```python
# COST_BUDGET env wins. Otherwise 80% of full-plan cost:
full_plan_cost = sum(m["cost"] for m in SKILL_REGISTRY.values()) + DEEP_TSFM_COST
return round(full_plan_cost * 0.8, 3)
```

This budget can trigger the skip-gate on heavier plans; activation is run/task dependent and can be overridden with `COST_BUDGET`.

---

## LLM provider routing

Planner and skill enrichment both route through the same
`_call_llm(system, user, max_tokens, ...)` helper in `skills.py`. Preferred
order is **watsonx → gemini → anthropic → groq**. The first provider to
return a non-empty string wins; failures fall through to the next one.
`LLM_PROVIDER` env reorders the chain.

Watsonx has extra resiliency:

- `WATSONX_MODEL_ID` accepts a **comma-separated list** of models tried in
  order, so a transient `downstream_request_failed` on one model falls
  through to the next (default: Llama-4 → Llama-3.3 → Mistral-Large).
- Exponential-backoff retry on 5xx errors (`WATSONX_MAX_RETRIES`, default 2).
- Parameters passed correctly to `chat()` with `generate_text()` fallback.

---

## Data sources

| Server | Default source | Fallback | Override env |
|---|---|---|---|
| IoT (sensor time series) | `IOT_CSV_DIR/*.csv` if set, else IoT MCP subprocess | mock deterministic generator | `USE_IOT_SUBPROCESS`, `IOT_CSV_DIR` |
| FMSR (failure mapping) | FMSR MCP subprocess via `uv run` | static failure-mode library | `USE_FMSR_SUBPROCESS` |
| TSFM (forecasting + deep TSAD) | TSFM MCP subprocess (`granite-tsfm`; needs PyTorch + GPU to be fast) | mock persistence forecast | `USE_TSFM_SUBPROCESS`, `PATH_TO_MODELS_DIR` |
| WO (work orders) | WO MCP subprocess with local CSV patch (reads `AssetOpsBench/src/tmp/assetopsbench/sample_data/*.csv`) | mock | `USE_WO_SUBPROCESS`, `WO_CSV_DIR` |

**WO does not need CouchDB.** The subprocess monkeypatches
`servers.wo.tools.load` to read the 4 bundled CSVs (249 rows each) directly.

**TSFM on CPU is too slow** for practical evaluation. Use Colab T4 (see
[Colab setup](#colab-setup)).

**IoT from `main.json`:** `scripts/extract_main_json.py` streams the
CouchDB export with `ijson` and writes one CSV per asset
(e.g. `data/chillers/chiller_6.csv`). Set `IOT_CSV_DIR` to that directory
and IoT calls read from CSVs — no CouchDB, no subprocess.

---

## Repo layout

```
SkillsAgent/
├── agent.py                    SkillAgent: planner + executor + DEEP_TSFM_COST
├── skills.py                   7 skills + SKILL_REGISTRY + _call_llm provider chain
├── knowledge.py                6 plugins + get_knowledge()
├── tools.py                    AssetOpsBench wrappers (IoT / FMSR / TSFM / WO) + mocks
├── confidence_evaluator.py     should_invoke_deep_tsfm(conf, theta)
├── eval_runner.py              Conditions A/B/C/D/F + Condition E θ sweep + Condition E cost_budget
├── trajectory_log.py           Append per-run JSONL trajectories
├── scenario_loader.py          12-task BUILTIN_TASK_BANK + HF ibm-research/AssetOpsBench loader
├── couch_export_catalog.py     Streaming merge of main.json → SensorMetadataPlugin catalog
├── run.py                      Quick benchmark on 4 scenarios
├── colab_setup.ipynb           Colab workflow (clone → extract → eval) on T4 GPU
├── scripts/
│   ├── calibrate_costs.py      Measure wall-clock latencies → skill_costs.json
│   ├── extract_main_json.py    main.json → data/chillers/<asset>.csv
│   ├── smoke_iot.py            IoT path live-check (subprocess or CSV)
│   ├── grade_assetops_metrics.py   Grade existing ablation CSVs with AssetOps evaluator
│   └── assetops_grader_worker.py   Worker invoked by grade_assetops_metrics via uv
├── tests/                      18 test modules (see §Tests)
├── eval_results/               ablation_results.csv + trajectories.jsonl per run
├── data/chillers/              Extracted per-asset CSVs (created by extract_main_json.py)
├── skill_costs.json            Calibrated per-skill median latency (optional)
├── .env / .env.public          Config (see §Environment variables)
└── requirements.txt
```

---

## Local setup

```bash
conda create -n skillsagent python=3.12 -y
conda activate skillsagent
pip install -r requirements.txt
```

Install [`uv`](https://docs.astral.sh/uv/) and initialise AssetOpsBench's
Python env so the MCP subprocesses can run:

```bash
brew install uv                                    # macOS
cd ../AssetOpsBench && uv sync && uv add granite-tsfm
cd ../SkillsAgent
```

For `AssetOpsBench/aobench`, prefer Python `<3.14` (3.12/3.13). The grader
stack depends on `pyarrow` via `mlflow`; on CPython 3.14 this may fall back to
source builds that fail without Arrow C++.

Copy and edit config:

```bash
cp .env.public .env
# Fill in WATSONX_API_KEY, WATSONX_PROJECT_ID, GEMINI_API_KEY, ANTHROPIC_API_KEY,
# GROQ_API_KEY (any subset — provider chain falls through), ASSETOPS path,
# COUCHDB_EXPORT_PATH (to your main.json), etc.
```

---

## Colab setup

TSFM on CPU is impractical (> 10 min per call). Use `colab_setup.ipynb`:

1. Upload `SkillsAgent/` and `main.json` to `MyDrive/HPML/project/` on Google
   Drive.
2. Open `colab_setup.ipynb` in Colab → `Runtime → Change runtime type → T4 GPU`.
3. Run cells top-to-bottom. The notebook will:
   - Mount Drive, then `git clone --depth=1` AssetOpsBench directly into
     `/content/work/` (fast — no multi-GB `rsync` from Drive).
   - `rsync` just the WO `sample_data/` and TSFM `tsfm_models/` into place.
   - Install requirements + `granite-tsfm` (torch + CUDA), pin
     `huggingface-hub<1.0` to satisfy `transformers 4.57`.
   - Run `scripts/extract_main_json.py` → `data/chillers/*.csv`.
   - Write a Colab `.env` with the right `PATH_TO_MODELS_DIR`, `IOT_CSV_DIR`,
     `WO_CSV_DIR`, `COUCHDB_EXPORT_PATH`.
   - Smoke-test `forecast_sensor` + `deep_tsfm_refine_anomalies` on T4.
   - Run `scripts/calibrate_costs.py` so `skill_costs.json` reflects GPU
     latency.
   - Run `eval_runner` → `eval_results/colab_<timestamp>/`.
   - Copy results back to Drive.

**Edit the Watsonx credentials** in the `.env`-writing cell before running
the smoke test, otherwise the planner will fall through the provider chain.

---

## Running the agent

```bash
python run.py
```

Runs four benchmark scenarios:

| Scenario | Typical plan |
|---|---|
| Why is Chiller 6 behaving abnormally? | `data_retrieval` → `anomaly_detection` → `root_cause_analysis` → `validate_failure` → `work_order_generation` |
| Forecast Chiller 6 condenser flow | `data_retrieval` → `forecasting` |
| Abnormal behavior in Chiller 6? | `data_retrieval` → `anomaly_detection` (early stop if no anomalies) |
| What sensors does Chiller 6 have? | `metadata_retrieval` |

---

## Running the ablation

```bash
python -m eval_runner --output-dir eval_results/local \
    --trajectory-log eval_results/local/trajectories.jsonl
```

Writes:
- `ablation_results.csv` — one row per (task × condition [× θ]) with
  `plan`, `skills_executed`, `skills_skipped`, `total_cost`, `latency_s`,
  `deep_tsfm_invoked`, `diagnosis_confidence_pre_deep`,
  `diagnosis_confidence`, `task_completion`.
- `trajectories.jsonl` — detailed per-run execution trace.

To run against the HF scenario bank:

```bash
python -m eval_runner --hf-limit 20 --output-dir eval_results/hf
```

Or from Python:

```python
from scenario_loader import load_hf_scenario_tasks
from eval_runner import evaluate_all

evaluate_all(output_dir="eval_results/hf",
             task_bank=load_hf_scenario_tasks(limit=20))
```

To run on `tsfm_report.csv` (same schema as the project root file: `id,type,label,text,tsfm_tools_called`):

```bash
python -m eval_runner --tsfm-report /path/to/tsfm_report.csv \
    --output-dir eval_results/tsfm_slice

# Optionally keep the builtin mini-bench first:
python -m eval_runner --tsfm-report ../tsfm_report.csv --prepend-builtin \
    --output-dir eval_results/combined
```

Alternatively set **`TSFM_REPORT_CSV`** to that path so you can omit `--tsfm-report`.

Categories are inferred per row (`Workorder` → fault_diagnosis; `FMSA` → fault_diagnosis; `multiagent` uses the tools column and a work-order phrase override — see `infer_tsfm_category` in `scenario_loader.py`).

To score ablation conditions with the AssetOpsBench evaluator (scenario server):

```bash
# 1) Start AssetOpsBench scenario server (default: http://localhost:8099)
# 2) Run scorer bridge from SkillsAgent
python scripts/score_with_assetopsbench.py \
    --scenario-set 13aab653-66fe-4fe6-84d8-89f1b18eede3 \
    --conditions C D F E \
    --output-dir eval_results/aob_tsfm_scored
```

To grade an existing `ablation_results.csv` with AssetOps criteria:

```bash
python scripts/grade_assetops_metrics.py \
    --ablation-csv eval_results/<run>/ablation_results.csv \
    --use-assetopsbench-rubrics \
    --aobench-root ../AssetOpsBench/aobench \
    --out-csv eval_results/<run>/assetops_metrics.csv \
    --pivot-csv eval_results/<run>/assetops_metrics_by_condition.csv
```

Useful scenario set IDs:
- General: `d3bec9b0-59b4-4a2f-9497-28cb1eed1c80`
- IoT: `b3aa206a-f7dc-43c9-a1f4-dcf984417487`
- TSFM: `13aab653-66fe-4fe6-84d8-89f1b18eede3`
- Workorders: `4021467f-363b-41d2-8c62-f6aa738b01b7`

Outputs:
- `grading_summary.csv` (accuracy per condition)
- `grading_details.csv` (per-scenario correctness + local run metrics)
- `graded_<condition>.json` (raw grader response with criteria details)

The default `BUILTIN_TASK_BANK` has **12 tasks** (4 fault-diagnosis, 3
forecasting, 3 anomaly-detection, 2 metadata) designed to vary in prompt
specificity so the `task_specificity` signal in
`score_diagnosis_confidence` produces per-task variance.

For a reproducible fetch → run → grade workflow against AssetOpsBench
scenario sets, see:

- `docs/assetopsbench_pipeline.md`

---

## Calibrating skill costs

```bash
python scripts/calibrate_costs.py --runs 3 --output skill_costs.json
```

Writes per-skill median wall-clock latency (plus `__deep_tsfm__`). The
executor picks this up automatically on next run. Re-calibrate whenever you
switch hardware (Mac CPU → Colab T4 etc.) so Condition E's budget reflects
real costs.

---

## Evaluation conditions

| Condition | Planning | Skills | Knowledge | Deep TSFM | Cost budget |
|---|---|---|---|---|---|
| **A** — raw LLM | — | — | — | — | — |
| **B** — tool baseline | — | tools only | — | — | — |
| **C** — planning only | ✅ | ✅ | ❌ (`KNOWLEDGE_INJECTION=0`) | — | — |
| **D** — skills + knowledge | ✅ | ✅ | ✅ | ❌ (`ENABLE_CONDITIONAL_DEEP_TSFM=0`) | — |
| **F** — skills + knowledge, always deep | ✅ | ✅ | ✅ | ✅ (no θ gate; `RCA_ALWAYS_DEEP_TSFM=1`) | — (no budget) |
| **E** — full system | ✅ | ✅ | ✅ | ✅ (θ-gated; `RCA_ALWAYS_DEEP_TSFM=0` in runner) | ✅ (80% of full-plan cost) |

**Comparing θ-gated vs “always run deep” in RCA:** use **E** (each θ) vs **F**. Both use the same skills, knowledge, and deep TSFM *machinery*; F bypasses `RCA_CONFIDENCE_THETA` and runs `deep_tsfm_refine_anomalies` whenever deep TSFM is enabled. Condition E also forces `RCA_ALWAYS_DEEP_TSFM=0` so a stray shell env does not break the sweep. For a **fair cost comparison** to F (no skips), run E with `COST_BUDGET=none` or override the budget env.

Condition E sweeps
`RCA_CONFIDENCE_THETA ∈ {0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95}` to expose
the tradeoff between deep-TSFM invocation rate and accuracy, straddling the
observed confidence knee (~0.6–0.65 on the default task bank).

---

## Results

Latest evaluated runs (AssetOpsBench grader outputs):

- `eval_results/colab_20260503_0230/assetops_metrics_by_condition.csv` — scenarios from `eval_inputs/tsfm_report/tsfm_report.json` (`n=54`)
- `eval_results/colab_20260503_2349/assetops_metrics_by_condition.csv` — scenarios from AssetOpsBench TSFM set `13aab653-66fe-4fe6-84d8-89f1b18eede3` (`n=23`); see `docs/assetopsbench_pipeline.md`

### Key condition summary (0230 vs 2349)

| Condition | 0230 `overall_correct` | 2349 `overall_correct` | 0230 `task_completion` | 2349 `task_completion` | 0230 `agent_sequence_correct` | 2349 `agent_sequence_correct` |
|---|---:|---:|---:|---:|---:|---:|
| **A — raw LLM** | 0.0000 | 0.1739 | 0.0652 | 0.1739 | 0.0652 | 0.3913 |
| **B — tool baseline** | 0.1304 | 0.0870 | 0.1957 | 0.2174 | 0.2609 | 0.1739 |
| **C — planning only** | 0.1087 | 0.2174 | 0.2667 | 0.2727 | 0.4444 | 0.3182 |
| **D — skills + knowledge (no deep)** | 0.2174 | 0.2174 | 0.4444 | 0.2857 | 0.8667 | 0.9524 |
| **E — best θ in each run** | **0.3043** (θ=0.8) | **0.2609** (θ=0.5/0.7/0.95) | **0.5556** (θ=0.8) | **0.3913** (θ=0.7/0.95) | 0.9091 (θ=0.9) | 0.9565 (θ=0.7) |
| **F — skills + knowledge, always deep** | 0.1739 | 0.2174 | 0.5333 | 0.3478 | **0.9333** | 0.9130 |

### What to take from these two runs

- Skill-aware pipelines (`D/E/F`) consistently dominate `A/B` on process metrics (`task_completion`, `agent_sequence_correct`), even when `overall_correct` is noisy.
- In `colab_20260503_0230`, **E at θ=0.8** is the best aggregate operating point (`overall_correct=0.3043`, `task_completion=0.5556`).
- In `colab_20260503_2349`, `overall_correct` is flatter across E settings (top `0.2609`), while `agent_sequence_correct` peaks at **θ=0.7** (`0.9565`).
- `hallucinations_rate` (lower is better) generally improves from `A` toward `D/E/F` in both runs, with the lowest observed values in E/F settings.
- Budget-cap skips are run-dependent: they appeared in `colab_20260503_0230` on heavy plans, but were not triggered in `colab_20260503_2349`.

---

## Environment variables

| Var | Purpose |
|---|---|
| `LLM_PROVIDER` | Preferred provider: `watsonx` / `gemini` / `anthropic` / `groq` |
| `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL` | IBM watsonx.ai credentials |
| `WATSONX_MODEL_ID` | Comma-separated fallback list (e.g. `meta-llama/llama-4-...,meta-llama/llama-3-3-70b-instruct`) |
| `WATSONX_MAX_RETRIES` | Retry count on 5xx (default 2) |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Google Gemini (default `gemini-2.5-flash`) |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | Anthropic Claude |
| `GROQ_API_KEY`, `GROQ_MODEL` | Groq (fast but rate-limited) |
| `ASSETOPS` | Path to `AssetOpsBench/src` |
| `PATH_TO_MODELS_DIR` | TSFM model checkpoints (e.g. `AssetOpsBench/src/tmp/tsfm_models`) |
| `USE_IOT_SUBPROCESS` / `USE_FMSR_SUBPROCESS` / `USE_TSFM_SUBPROCESS` / `USE_WO_SUBPROCESS` | Toggle each AssetOpsBench subprocess (`1` default) |
| `IOT_CSV_DIR` | Per-asset CSV directory (bypasses CouchDB) |
| `WO_CSV_DIR` | WO sample CSVs (auto-detected from `ASSETOPS`) |
| `COUCHDB_EXPORT_PATH` | Path to `main.json` — enables `SensorMetadataPlugin` live merge |
| `RCA_CONFIDENCE_THETA` | Deep-TSFM gate threshold (0 ≤ θ ≤ 1; default 0.85) |
| `RCA_ALWAYS_DEEP_TSFM` | `1` = always run deep TSFM in RCA when enabled (ablation F); θ ignored |
| `ENABLE_CONDITIONAL_DEEP_TSFM` | `0` disables the gate (Condition D) |
| `KNOWLEDGE_INJECTION` | `0` disables all plugins (Condition C) |
| `SKILL_COSTS_PATH` | Path to calibrated `skill_costs.json` (default `./skill_costs.json`) |
| `DEEP_TSFM_COST` | Override cost charged on deep-TSFM invocation |
| `COST_BUDGET` | Override Condition E budget (float, or `none`) |
| `TSFM_CATALOG_INJECTION` | `0` disables TSFM static task/model injection in `metadata_retrieval` |
| `TSFM_REPORT_CSV` | Path to `tsfm_report.csv` — `eval_runner` uses it when `--tsfm-report` is omitted |
| `TRAJECTORY_LOG_PATH` | Append per-run JSONL trajectories to this file |
| `TRACE_VERBOSE` | `1` adds redacted per-skill `context_before`/`context_after` snapshots to `metrics.skill_steps` |

---

## Tests

```bash
python -m pytest tests/ -v
```

Current suite covers 18 test modules:

| Module | What it covers |
|---|---|
| `test_confidence_evaluator.py` | `should_invoke_deep_tsfm` gate, θ env var, enable/disable toggle |
| `test_cost_budget.py` | Calibrated costs override priors + Condition E skip-on-budget |
| `test_couch_export.py` | `couch_export_catalog` streaming parse of `main.json` |
| `test_deep_tsfm_cost.py` | `DEEP_TSFM_COST` charged iff `deep_tsfm_invoked` |
| `test_eval_runner.py` | Conditions B/C/D, θ sweep coverage, `_task_completion_score` |
| `test_grade_assetops_metrics.py` | Rubric merging + `TSFM_*` task-id normalization for grader payloads |
| `test_graded_confidence.py` | All six confidence signals, including `task_specificity` |
| `test_iot_csv_fallback.py` | `IOT_CSV_DIR` path bypasses the subprocess |
| `test_knowledge.py` | Plugin routing + `KNOWLEDGE_INJECTION=0` disables injection |
| `test_llm_provider.py` | Provider routing + watsonx model fallback |
| `test_scenario_loader.py` | `BUILTIN_TASK_BANK` shape and category coverage |
| `test_skills_rca.py` | RCA end-to-end with/without deep TSFM |
| `test_tsfm_catalog.py` | TSFM catalog parity (`servers.tsfm.models` static task/model lists) |
| `test_tsfm_task_spec.py` | Parsing official TSFM forecast prompts + dataset-path resolution |
| `test_tools_smoke.py` | Mock-path smoke test for each `tools.py` entry point |
| `test_trajectory_log.py` | JSONL trajectory writer |
| `test_wo_local_csv.py` | WO real Markov predictions via local CSV (skipped if `ASSETOPS` unset) |
