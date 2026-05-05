"""Optional Weights & Biases integration for ``eval_runner`` ablation runs.

Enable by setting ``WANDB_PROJECT`` (and ``WANDB_API_KEY`` in Colab secrets or env).
Disable with ``WANDB_DISABLED=1``.

Env (all optional except project when you want logging):
  WANDB_PROJECT       — required to turn logging on (default: off)
  WANDB_ENTITY        — team/entity name
  WANDB_RUN_GROUP     — group runs in the UI (e.g. ``colab_20260503``)
  WANDB_RUN_NAME      — single-run display name
  WANDB_LOG_ARTIFACT  — ``1`` to upload ``ablation_results.csv`` at the end
  WANDB_DISABLED      — ``1`` to force no-op even if ``wandb`` is installed
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_step: int = 0


def wandb_eval_enabled() -> bool:
    if os.environ.get("WANDB_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        return False
    if not (os.environ.get("WANDB_PROJECT") or "").strip():
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.warning("wandb not installed; pip install wandb or set WANDB_DISABLED=1")
        return False
    return True


def wandb_eval_init(config: dict[str, Any]) -> None:
    """Start a run. Safe no-op if disabled or init fails."""
    global _step
    _step = 0
    if not wandb_eval_enabled():
        return
    import wandb

    if wandb.run is not None:
        wandb.finish()
    project = os.environ["WANDB_PROJECT"].strip()
    kwargs: dict[str, Any] = {
        "project": project,
        "config": config,
        "reinit": True,
    }
    ent = os.environ.get("WANDB_ENTITY", "").strip()
    if ent:
        kwargs["entity"] = ent
    grp = os.environ.get("WANDB_RUN_GROUP", "").strip()
    if grp:
        kwargs["group"] = grp
    name = os.environ.get("WANDB_RUN_NAME", "").strip()
    if name:
        kwargs["name"] = name
    tags = [t.strip() for t in os.environ.get("WANDB_TAGS", "").split(",") if t.strip()]
    if tags:
        kwargs["tags"] = tags
    try:
        wandb.init(**kwargs)
    except Exception as e:
        logger.warning("wandb.init failed (%s); continuing without W&B", e)


def _as_float(x: Any, default: float | None = None) -> float | None:
    if x in ("", None):
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def wandb_eval_log_row(row: dict[str, Any]) -> None:
    """Log one eval row (one task × one condition)."""
    global _step
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    _step += 1
    err = row.get("error") or ""
    payload: dict[str, Any] = {
        "condition": str(row.get("condition", "")),
        "theta": str(row.get("theta", "")),
        "category": str(row.get("category", "")),
        "task_id": str(row.get("task_id", "")),
        "task_completion": _as_float(row.get("task_completion")),
        "total_cost": _as_float(row.get("total_cost")),
        "latency_s": _as_float(row.get("latency_s")),
        "tool_calls": _as_float(row.get("tool_calls")),
        "skills_skipped": _as_float(row.get("skills_skipped")),
        "skipped_conditional": _as_float(row.get("skipped_conditional")),
        "run_failed": bool(err),
    }
    dti = row.get("deep_tsfm_invoked")
    if isinstance(dti, bool):
        payload["deep_tsfm_invoked"] = float(dti)
    elif dti not in ("", None):
        payload["deep_tsfm_invoked"] = 1.0 if str(dti).lower() in ("true", "1", "yes") else 0.0
    wandb.log({k: v for k, v in payload.items() if v is not None}, step=_step)


def wandb_eval_finish(csv_path: Path | None = None) -> None:
    """Upload optional artifact and end run."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    try:
        if csv_path and csv_path.is_file():
            if os.environ.get("WANDB_LOG_ARTIFACT", "").strip().lower() in ("1", "true", "yes"):
                art = wandb.Artifact("ablation_results", type="dataset")
                art.add_file(str(csv_path))
                wandb.log_artifact(art)
    except Exception as e:
        logger.warning("wandb artifact upload failed: %s", e)
    finally:
        try:
            wandb.finish()
        except Exception:
            pass
