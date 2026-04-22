"""Scenario sources for evaluation (proposal Phases 1 / 6).

- ``BUILTIN_TASK_BANK``: small local set for fast ablations (no network).
- ``load_hf_scenario_tasks``: optional full AssetOpsBench split via Hugging Face
  ``ibm-research/AssetOpsBench`` (requires ``datasets`` + auth as needed).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# (task_id, prompt, category) — categories must match ``_EXPECTED_SKILLS`` /
# ``_FINAL_KEYS`` in ``eval_runner``. Prompts intentionally vary in specificity
# (vague → keyword-rich) so the ``task_specificity`` term in
# ``score_diagnosis_confidence`` produces per-task variance and the θ sweep
# yields a graded cost/accuracy curve rather than a step at one bucket edge.
BUILTIN_TASK_BANK: list[tuple[str, str, str]] = [
    ("T01", "Why is Chiller 6 behaving abnormally and do we need a work order?", "fault_diagnosis"),
    ("T02", "Forecast next week's condenser water flow for Chiller 9.", "forecasting"),
    ("T03", "Was there any abnormal behavior in Chiller 9 over the past week?", "anomaly_detection"),
    ("T04", "What sensors are available for Chiller 6, and what do they measure?", "metadata"),
    ("T05", "Chiller 9 vibration has been rising — should we schedule service?", "fault_diagnosis"),
    ("T06", "Predict COP for Chiller 6 next month and flag if maintenance needed.", "forecasting"),
    ("T07", "Chiller 6 refrigerant pressure is dropping and evaporator temp is off — diagnose and open a work order if needed.", "fault_diagnosis"),
    ("T08", "Chiller 9 compressor power draw has spiked; explain the root cause.", "fault_diagnosis"),
    ("T09", "Something feels off with Chiller 6 — can you look into it?", "fault_diagnosis"),
    ("T10", "Project Chiller 9 compressor speed over the next 3 days.", "forecasting"),
    ("T11", "Check Chiller 6 chilled-water supply temperature for anomalies this week.", "anomaly_detection"),
    ("T12", "List the sensors instrumented on Chiller 9 with their units.", "metadata"),
]


def load_hf_scenario_tasks(
    *,
    split: str = "train",
    limit: int | None = None,
) -> list[tuple[str, str, str]]:
    """Load (id, text, category) rows from the HF scenarios config, or fall back to builtin.

    Schema varies by benchmark version; we accept common field names.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets not installed — using BUILTIN_TASK_BANK")
        return list(BUILTIN_TASK_BANK)

    try:
        ds = load_dataset("ibm-research/AssetOpsBench", "scenarios", trust_remote_code=True)
    except Exception as e:
        logger.warning("HF scenario load failed (%s) — using BUILTIN_TASK_BANK", e)
        return list(BUILTIN_TASK_BANK)

    if split not in ds:
        logger.warning("split %r not in dataset keys %s — using BUILTIN_TASK_BANK", split, list(ds.keys()))
        return list(BUILTIN_TASK_BANK)

    table = ds[split]
    rows: list[tuple[str, str, str]] = []
    n = len(table)
    cap = n if limit is None else min(n, limit)
    for i in range(cap):
        row: dict[str, Any] = table[i]
        sid = str(row.get("id", f"HF_{i}"))
        text = (
            row.get("text")
            or row.get("utterance")
            or row.get("question")
            or row.get("prompt")
            or ""
        )
        if not text:
            continue
        cat = row.get("type") or row.get("category") or row.get("domain") or "unknown"
        rows.append((sid, str(text), str(cat)))

    if not rows:
        logger.warning("HF split produced no rows — using BUILTIN_TASK_BANK")
        return list(BUILTIN_TASK_BANK)

    return rows
