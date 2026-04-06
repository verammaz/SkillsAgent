# SkillsAgent — Setup & Run Guide

Skill-augmented, knowledge-aware agent for AssetOpsBench fault diagnosis tasks.

---

## Prerequisites

- Python 3.12+
- Conda (recommended) or any virtual environment manager
- [`uv`](https://docs.astral.sh/uv/) — required to call the AssetOpsBench FMSR server
  ```bash
  brew install uv   # macOS
  ```
- The **AssetOpsBench** repo cloned and its dependencies installed via `uv sync`

---

## 1. Create and activate the environment

```bash
conda create -n skillsagent python=3.12
conda activate skillsagent
pip install -r requirements.txt
```

---

## 2. Configure environment variables

Copy the template and fill in your keys:

```bash
cp .env.public .env
```

Edit `.env`:

```bash
# LLM for planning and enrichment (choose one)
LLM_PROVIDER=groq                          # or: anthropic
GROQ_API_KEY=gsk_...                       # free at console.groq.com
ANTHROPIC_API_KEY=sk-ant-...               # optional; used if LLM_PROVIDER=anthropic

# For FMSR MCP server (via AssetOpsBench uv env)
LITELLM_API_KEY=sk-proj-...                # OpenAI key used by FMSR LLM fallback
LITELLM_BASE_URL=https://api.openai.com/v1
FMSR_MODEL_ID=openai/gpt-4o-mini

# Path to the AssetOpsBench source directory
ASSETOPS=/path/to/AssetOpsBench/src
```

> **Note:** For chiller/AHU queries the FMSR server uses a curated local YAML file —
> `LITELLM_API_KEY` is only needed if querying unknown asset types.

---

## 3. Run

```bash
# Make sure run.py imports from agent (not deep_agent)
# Line 12 should read: from agent import SkillAgent

python run.py
```

This runs the four benchmark scenarios:

| Scenario | Skills executed |
|---|---|
| Why is Chiller 6 behaving abnormally? | data_retrieval → anomaly_detection → root_cause_analysis → validate_failure → work_order_generation |
| Forecast Chiller 9 condenser flow | data_retrieval → forecasting |
| Abnormal behavior in Chiller 9? | data_retrieval → anomaly_detection (early stop if no anomalies) |
| What sensors does Chiller 6 have? | metadata_retrieval |

Results are printed to stdout. To save: `python run.py > result.txt`.

---

## 4. Run the ablation evaluation

```bash
python eval_runner.py
# results → eval_results/ablation_results.csv
```

Runs four conditions (A: baseline LLM, B: +planning, C: +planning+skills, D: full system) across all tasks.

---

## Files

```
run.py
agent.py          SkillAgent: LLM planner + conditional executor
skills.py   7 skill functions (data retrieval, anomaly detection, RCA, ...)
knowledge.py  5 knowledge plugins (sensor metadata, failure modes, policy, ...)
tools.py    Agent wrappers:
                  IoT / TSFM / WO  →  mock (real agents coming)
                  FMSR             →  real (via uv subprocess → AssetOpsBench)
```

## Agent Selection

`run.py` has two import options — uncomment the one you want:

```python
from agent import SkillAgent       # Vera's agent (this repo)
# from deep_agent import SkillAgent  # Mana's agent
```
