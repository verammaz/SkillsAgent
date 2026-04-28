"""Scenario sources for evaluation (proposal Phases 1 / 6).

- ``BUILTIN_TASK_BANK``: small local set for fast ablations (no network).
- ``load_hf_scenario_tasks``: optional full AssetOpsBench split via Hugging Face
  ``ibm-research/AssetOpsBench`` (requires ``datasets`` + auth as needed).
- ``load_tsfm_report_tasks``: curated scenarios from ``tsfm_report.csv`` (TSFM paper / bench slice).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def infer_tsfm_category(row: dict[str, str]) -> str:
    """Map a ``tsfm_report.csv`` row to ``eval_runner`` categories.

    ``Workorder`` / ``FMSA`` rows route to fault_diagnosis (WO + RCA-heavy).
    ``multiagent`` rows use ``tsfm_tools_called`` gold tools where possible:
      forecasting-only → forecasting; TSAD-only → anomaly_detection;
      mixed tools → fault_diagnosis.
    Overrides to fault_diagnosis when the prompt asks for work-order decisions.
    """
    text_l = (row.get("text") or "").lower()
    row_type = (row.get("type") or "").strip()

    if row_type in {"Workorder", "FMSA"}:
        return "fault_diagnosis"

    wo_markers = ("work order", "corrective work order", "preventive work orders")
    if row_type == "multiagent" and any(m in text_l for m in wo_markers):
        return "fault_diagnosis"

    tools_l = (row.get("tsfm_tools_called") or "").lower()
    has_fcst = "run_tsfm_forecasting" in tools_l
    has_tsad = "run_integrated_tsad" in tools_l or "run_tsad" in tools_l

    if has_fcst and not has_tsad:
        return "forecasting"
    if has_tsad and not has_fcst:
        return "anomaly_detection"
    return "fault_diagnosis"


def load_tsfm_report_tasks(csv_path: str | Path) -> list[tuple[str, str, str]]:
    """Load ``(task_id, prompt, category)`` tuples from ``tsfm_report.csv``.

    Columns: ``id,type,label,text,tsfm_tools_called``. BOM-safe UTF-8.
    Task IDs are prefixed ``TSFM_<id>`` to avoid collisions with builtin IDs.
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        logger.warning("tsfm_report CSV not found: %s — returning empty bank", path)
        return []

    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            logger.warning("tsfm_report CSV missing required columns: %s", path)
            return []
        id_key = "id" if "id" in reader.fieldnames else reader.fieldnames[0]
        for r in reader:
            if not isinstance(r, dict):
                continue
            text = (r.get("text") or "").strip()
            if not text:
                continue
            raw_id = str((r.get(id_key) or "")).strip() or str(len(rows))
            tid = f"TSFM_{raw_id}"
            if tid in seen:
                tid = f"TSFM_{raw_id}_{len(rows)}"
            seen.add(tid)
            cat = infer_tsfm_category({k: (v or "") for k, v in r.items()})
            rows.append((tid, text, cat))

    if not rows:
        logger.warning("tsfm_report CSV produced no rows: %s", path)
    else:
        logger.info("Loaded %d tasks from %s", len(rows), path)
    return rows


def default_tsfm_report_path() -> Path | None:
    """``<repo>/tsfm_report.csv`` next to the ``SkillsAgent/`` package, if present."""
    candidate = Path(__file__).resolve().parent.parent / "tsfm_report.csv"
    return candidate if candidate.is_file() else None

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
