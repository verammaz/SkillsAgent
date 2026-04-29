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
13. [Using the DeepAgents agent](#using-the-deepagents-agent)
14. [Running the ablation](#running-the-ablation)
15. [Calibrating skill costs](#calibrating-skill-costs)
16. [Evaluation conditions](#evaluation-conditions)
17. [Results](#results)
18. [Environment variables](#environment-variables)
19. [Tests](#tests)

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
| 2 | `metadata_retrieval` | `get_asset_metadata` → IoT MCP | — (LLM summarizes raw) | never | always (self-contained) |
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

This budget is tight enough to force the skip-gate to fire on heavy plans.

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
├── deep_agent.py               DeepAgents-based orchestrator (LangChain/LangGraph)
├── skills.py                   7 skills + SKILL_REGISTRY + _call_llm provider chain
├── knowledge.py                6 plugins + get_knowledge()
├── tools.py                    AssetOpsBench wrappers (IoT / FMSR / TSFM / WO) + mocks
├── confidence_evaluator.py     should_invoke_deep_tsfm(conf, theta)
├── eval_runner.py              Conditions A–E, θ sweep, Condition E cost_budget
├── trajectory_log.py           Append per-run JSONL trajectories
├── scenario_loader.py          12-task BUILTIN_TASK_BANK + HF ibm-research/AssetOpsBench loader
├── couch_export_catalog.py     Streaming merge of main.json → SensorMetadataPlugin catalog
├── run.py                      Quick benchmark on 4 scenarios
├── colab_setup.ipynb           Colab workflow (clone → extract → eval) on T4 GPU
├── scripts/
│   ├── calibrate_costs.py      Measure wall-clock latencies → skill_costs.json
│   ├── extract_main_json.py    main.json → data/chillers/<asset>.csv
│   └── smoke_iot.py            IoT path live-check (subprocess or CSV)
├── tests/                      15 test modules (see §Tests)
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

## Using the DeepAgents agent

`deep_agent.py` is an alternative orchestration layer for the same AssetOpsBench
toolset, built on top of [LangChain Deep Agents](https://github.com/langchain-ai/deepagents)
(`create_deep_agent` / LangGraph) instead of the custom planner/executor in `agent.py`.

### How it differs from `agent.py` / `run.py`

| | `agent.py` (skills-based) | `deep_agent.py` (DeepAgents-based) |
|---|---|---|
| Orchestration | Custom planner + skill executor | `create_deep_agent` (LangGraph ReAct) |
| Tool registry | `SKILL_REGISTRY` in `skills.py` | 6 `@tool`-decorated callables |
| Conditional deep TSFM | Inside `root_cause_analysis` skill | **Inside** `fmsr_root_cause_tool` (deterministic Python gate, not delegated to the LLM) |
| Knowledge plugins | Called per-skill from `knowledge.py` | Called inside each tool body |
| Cost budget | `cost_budget` arg to `SkillAgent` | Tracked in metrics; not enforced as a hard gate |
| Tracing | `skills_executed`, `skipped_conditional`, `skipped_early_stop` | Same keys in `metrics`, inferred from tool-call trace |
| Entry point | `SkillAgent().run(task)` | `SkillAgent().run(task)` — same interface |

Both agents use the **same** real tool wrappers (`tools.py`) and the **same**
LLM provider fallback chain (watsonx → gemini → anthropic → groq).

### Additional setup

Install the DeepAgents orchestration layer and its LangChain provider packages:

```bash
pip install deepagents langchain-ibm langchain-google-genai langchain-anthropic langchain-groq
```

> **LLM credentials** — `deep_agent.py` uses the same `.env` as the rest of
> the project. At least one of the following must be set:
>
> | Provider | Required env vars |
> |---|---|
> | IBM watsonx (default) | `WATSONX_API_KEY`, `WATSONX_PROJECT_ID` |
> | Google Gemini | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
> | Anthropic Claude | `ANTHROPIC_API_KEY` |
> | Groq | `GROQ_API_KEY` |
>
> Set `LLM_PROVIDER=gemini` (or `anthropic` / `groq`) to skip watsonx.

### Run the smoke test (no live backend required)

```bash
python tests/test_deep_agent_smoke.py
```

Expected output (all 8 pass without AssetOpsBench or an API key — tools use
mock fallbacks, and the agent is stubbed):

```
  PASS  test_iot_data_retrieval_tool
  PASS  test_sensor_metadata_tool
  PASS  test_lightweight_anomaly_tool
  PASS  test_fmsr_root_cause_tool
  PASS  test_forecasting_tool
  PASS  test_work_order_tool
  PASS  test_run_deep_agent_metadata_scenario
  PASS  test_skill_agent_run
```

### Run a fault-diagnosis query

```python
from deep_agent import SkillAgent

agent = SkillAgent()
out = agent.run("Why is Chiller 6 behaving abnormally, and do we need a work order?")
print(out)
```

Or with the convenience one-liner:

```python
from deep_agent import run_deep_agent

out = run_deep_agent(
    "Why is Chiller 6 behaving abnormally, and do we need a work order?",
    threshold=0.8,   # RCA confidence threshold for deep TSFM (overrides env var)
)
print(out["result"]["answer"])
```

From the shell:

```bash
python - <<'PY'
from deep_agent import SkillAgent
agent = SkillAgent()
out = agent.run("Why is Chiller 6 behaving abnormally, and do we need a work order?")
import json; print(json.dumps(out, indent=2, default=str))
PY
```

### Supported scenario types

| Scenario | Example query | Key tools called |
|---|---|---|
| Fault diagnosis + work order | "Why is Chiller 6 behaving abnormally, and do we need a work order?" | `lightweight_anomaly_tool` → `fmsr_root_cause_tool` → `work_order_tool` |
| Anomaly detection only | "Was there any abnormal behavior in Chiller 9 over the past week?" | `iot_data_retrieval_tool` → `lightweight_anomaly_tool` |
| Forecasting + preventive maintenance | "Forecast next week's condenser water flow for Chiller 9." | `forecasting_tool` (→ `work_order_tool` if breach detected) |
| Sensor metadata | "What sensors are available for Chiller 6, and what do they measure?" | `sensor_metadata_tool` |

### Output format

`SkillAgent.run()` returns a dict with two top-level keys:

```python
{
  "result": {
    "answer": "<LLM final answer text>",
    "failure": "<diagnosed failure mode or None>",
    "anomaly_analysis": { ... },   # from lightweight_anomaly_tool
    "work_order": "<WO-ID or None>",
    "forecast": { ... },           # from forecasting_tool
    "metadata": [ ... ],           # from sensor_metadata_tool
    "raw": { ... },                # merged payload from all tool outputs
  },
  "metrics": {
    "plan": ["anomaly_detection", "root_cause_analysis", ...],  # skill names called
    "skills_executed": [ ... ],
    "skills_skipped": [ ... ],
    "skipped_conditional": [],
    "skipped_early_stop": [ ... ],
    "stopped_at": "<last skill name>",
    "tool_calls": 3,
    "total_cost": 1.7,
    "latency_s": 4.2,
    "deep_tsfm_invoked": False,
    "diagnosis_confidence": 0.712,
    "diagnosis_confidence_pre_deep": 0.573,
    "confidence": 0.712,
    "tsfm_deep_invoked": False,
  }
}
```

The `metrics` dict is fully compatible with `eval_runner.py`'s `_append_row`
and `run.py`'s `_print_metrics` — the DeepAgents agent can be dropped in
anywhere the skills-based agent is used.


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

The default `BUILTIN_TASK_BANK` has **12 tasks** (4 fault-diagnosis, 3
forecasting, 3 anomaly-detection, 2 metadata) designed to vary in prompt
specificity so the `task_specificity` signal in
`score_diagnosis_confidence` produces per-task variance.

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
| **E** — full system | ✅ | ✅ | ✅ | ✅ (θ-gated) | ✅ (80% of full-plan cost) |

Condition E sweeps
`RCA_CONFIDENCE_THETA ∈ {0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95}` to expose
the tradeoff between deep-TSFM invocation rate and accuracy, straddling the
observed confidence knee (~0.6–0.65 on the default task bank).

---

## Results

Latest ablation run: `eval_results/colab_20260422_1701/ablation_results.csv`
— 12 tasks × 11 conditions = 132 rows, executed on a Colab T4 with real
watsonx (Llama-4), real IoT CSVs from `main.json`, real TSFM (`ttm_96_28`),
real WO CSVs, and calibrated skill costs.

### Condition summary (mean across 12 tasks)

| Condition | `task_completion` | `total_cost` | `latency_s` | deep TSFM % | tool calls |
|---|---:|---:|---:|---:|---:|
| **A — raw LLM** | 0.222 | 0.00 | 4.7 | 0% | 1.0 |
| **B — tool baseline** | 0.278 | 0.85 | 3.8 | 0% | 3.4 |
| **C — planning only** | 0.958 | 26.81 | 14.9 | 0% | 3.3 |
| **D — skills + knowledge** | 0.931 | 25.29 | 14.9 | 0% | 3.3 |
| **E, θ = 0.50** | 0.931 | 24.74 | 14.2 | 0% | 3.2 |
| **E, θ = 0.60** | 0.931 | 29.54 | 18.3 | 25% | 3.3 |
| **E, θ = 0.65** | 0.931 | 33.16 | 22.1 | 50% | 3.1 |
| **E, θ = 0.70** | 0.958 | 38.07 | 26.9 | 67% | 3.2 |
| **E, θ = 0.80** | 0.958 | 38.07 | 24.4 | 67% | 3.2 |
| **E, θ = 0.90** | 0.931 | 34.58 | 22.7 | 58% | 3.1 |
| **E, θ = 0.95** | 0.958 | 38.07 | 25.5 | 67% | 3.2 |

**Take-aways.**

- Skill-aware pipelines (C / D / E) lift `task_completion` by **~4x** over
  the raw-LLM (A) and static-tool (B) baselines.
- The θ sweep produces the expected monotone cost-for-deep-invocation curve:
  0 / 25% / 50% / 67% deep TSFM as θ moves from 0.50 → 0.70. Plateau above
  0.70 reflects the task bank's max pre-deep confidence (0.657); θ = 0.95
  just stress-tests the gate.
- Condition D vs E at θ ≥ 0.70 differ by ~12 cost units — the
  `DEEP_TSFM_COST` + extra RCA latency the gate trades for accuracy.

### Graded pre-deep confidence (Condition D, per task)

Prompts vary from vague to keyword-rich; the `task_specificity` signal
produces four distinct pre-deep confidence buckets on the default bank:

| Tasks | Prompt style | `diagnosis_confidence_pre_deep` |
|---|---|---:|
| T01, T03, T09 | Vague (“behaving abnormally”, “was there any abnormal”, “something feels off”) | 0.573 |
| T05, T06, T11 | Single keyword (“vibration”, “COP”, “chilled-water”) | 0.603 |
| T08 | Two keywords (“compressor power draw”) | 0.621 |
| T07 | Subsystem-specific (“refrigerant pressure … evaporator temp”) | 0.657 |

### Post-deep confidence lift (Condition E, θ = 0.70)

When the gate fires, deep TSFM reliably lifts confidence by ~0.05–0.12 via
the TSAD-record corroboration signal:

| Task | pre-deep | post-deep | Δ |
|---|---:|---:|---:|
| T01 | 0.573 | 0.690 | +0.117 |
| T03 | 0.573 | 0.625 | +0.052 |
| T05 | 0.603 | 0.655 | +0.052 |
| T06 | 0.603 | 0.720 | +0.117 |
| T07 | 0.657 | 0.774 | +0.117 |
| T08 | 0.621 | 0.673 | +0.052 |
| T09 | 0.573 | 0.690 | +0.117 |
| T11 | 0.603 | 0.720 | +0.117 |

### Cost budgeting

Condition E's 80%-of-full-plan budget activates on heavy plans: on T06
(forecasting + fault-diagnosis plan with 6 skills), `validate_failure` and
`work_order_generation` are skipped at θ ≥ 0.65 (`skipped_conditional = 2`)
while `task_completion` stays at 1.0 — the agent correctly drops the
non-terminal policy check without losing the deliverable.

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
| `ENABLE_CONDITIONAL_DEEP_TSFM` | `0` disables the gate (Condition D) |
| `KNOWLEDGE_INJECTION` | `0` disables all plugins (Condition C) |
| `SKILL_COSTS_PATH` | Path to calibrated `skill_costs.json` (default `./skill_costs.json`) |
| `DEEP_TSFM_COST` | Override cost charged on deep-TSFM invocation |
| `COST_BUDGET` | Override Condition E budget (float, or `none`) |
| `TRAJECTORY_LOG_PATH` | Append per-run JSONL trajectories to this file |

---

## Tests

```bash
python -m pytest tests/ -v
```

68 tests across 14 modules:

| Module | What it covers |
|---|---|
| `test_confidence_evaluator.py` | `should_invoke_deep_tsfm` gate, θ env var, enable/disable toggle |
| `test_cost_budget.py` | Calibrated costs override priors + Condition E skip-on-budget |
| `test_couch_export.py` | `couch_export_catalog` streaming parse of `main.json` |
| `test_deep_tsfm_cost.py` | `DEEP_TSFM_COST` charged iff `deep_tsfm_invoked` |
| `test_eval_runner.py` | Conditions B/C/D, θ sweep coverage, `_task_completion_score` |
| `test_graded_confidence.py` | All six confidence signals, including `task_specificity` |
| `test_iot_csv_fallback.py` | `IOT_CSV_DIR` path bypasses the subprocess |
| `test_knowledge.py` | Plugin routing + `KNOWLEDGE_INJECTION=0` disables injection |
| `test_llm_provider.py` | Provider routing + watsonx model fallback |
| `test_scenario_loader.py` | `BUILTIN_TASK_BANK` shape and category coverage |
| `test_skills_rca.py` | RCA end-to-end with/without deep TSFM |
| `test_tools_smoke.py` | Mock-path smoke test for each `tools.py` entry point |
| `test_trajectory_log.py` | JSONL trajectory writer |
| `test_wo_local_csv.py` | WO real Markov predictions via local CSV (skipped if `ASSETOPS` unset) |
| `test_deep_agent_smoke.py` | 6 tool-level + 2 agent-level tests for `deep_agent.py` (no live backend needed) |
