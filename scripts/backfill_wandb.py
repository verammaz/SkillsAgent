#!/usr/bin/env python3
"""Backfill historical ``eval_results/*`` folders into W&B (one folder = one run).

Usage example::

    cd SkillsAgent
    WANDB_PROJECT=my-project WANDB_ENTITY=my-team \\
    python scripts/backfill_wandb.py \\
      --eval-root eval_results \\
      --group historical-backfill \\
      --artifact

Notes:
  - Requires ``WANDB_PROJECT`` (unless ``--dry-run``).
  - Uses ``wandb_tracking.py`` so payload shape matches live ``eval_runner`` logs.
  - Prefers ``ablation_results.csv``; falls back to ``assetops_metrics.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

# Allow ``python scripts/backfill_wandb.py`` without PYTHONPATH (repo root must be importable).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_rs = str(_REPO_ROOT)
if _rs not in sys.path:
    sys.path.insert(0, _rs)


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_eval_dirs(eval_root: Path, run_names: set[str] | None) -> list[Path]:
    if not eval_root.is_dir():
        return []
    dirs = [p for p in eval_root.iterdir() if p.is_dir()]
    if run_names:
        dirs = [p for p in dirs if p.name in run_names]
    return sorted(dirs, key=lambda p: p.name)


def _pick_csv(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "ablation_results.csv",
        run_dir / "assetops_metrics.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _count_rows(path: Path) -> int:
    csv.field_size_limit(min(2**31 - 1, 10_000_000))
    with open(path, newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    csv.field_size_limit(min(2**31 - 1, 10_000_000))
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return (reader.fieldnames or []), rows


def _as_float(x: Any) -> float | None:
    if x in ("", None):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _log_assetops_tables(run_dir: Path) -> None:
    """Attach `assetops_metrics*.csv` as W&B tables and log compact summaries."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    metrics_csv = run_dir / "assetops_metrics.csv"
    by_cond_csv = run_dir / "assetops_metrics_by_condition.csv"

    if metrics_csv.is_file():
        cols, rows = _read_csv_rows(metrics_csv)
        if cols and rows:
            table = wandb.Table(columns=cols, data=[[r.get(c, "") for c in cols] for r in rows])
            wandb.log({"assetops_metrics_table": table})

    if by_cond_csv.is_file():
        cols, rows = _read_csv_rows(by_cond_csv)
        if cols and rows:
            table = wandb.Table(columns=cols, data=[[r.get(c, "") for c in cols] for r in rows])
            wandb.log({"assetops_metrics_by_condition_table": table})

            # Helpful scalar summary for dashboard sorting/filtering.
            best = max(rows, key=lambda r: _as_float(r.get("overall_correct_rate")) or -1.0)
            wandb.log(
                {
                    "assetops_best_condition": str(best.get("condition", "")),
                    "assetops_best_theta": str(best.get("theta", "")),
                    "assetops_best_overall_correct_rate": _as_float(best.get("overall_correct_rate")),
                    "assetops_best_task_completion_rate": _as_float(best.get("task_completion_rate")),
                    "assetops_best_agent_sequence_correct_rate": _as_float(
                        best.get("agent_sequence_correct_rate")
                    ),
                }
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="Directory containing run folders (default: SkillsAgent/eval_results)",
    )
    ap.add_argument(
        "--runs",
        type=str,
        default="",
        help="Comma-separated run folder names to backfill (default: all subdirs)",
    )
    ap.add_argument(
        "--group",
        type=str,
        default="historical-backfill",
        help="W&B run group for all uploaded runs",
    )
    ap.add_argument(
        "--artifact",
        action="store_true",
        help="Upload each source CSV as W&B artifact (sets WANDB_LOG_ARTIFACT=1)",
    )
    ap.add_argument(
        "--log-tables",
        action="store_true",
        default=True,
        help="When present files exist, log assetops_metrics*.csv as W&B tables (default: on)",
    )
    ap.add_argument(
        "--no-log-tables",
        dest="log_tables",
        action="store_false",
        help="Disable W&B table logging for assetops metrics CSVs",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without calling W&B",
    )
    args = ap.parse_args()

    root = _skills_root()
    eval_root = args.eval_root.expanduser().resolve() if args.eval_root else (root / "eval_results")
    run_names = {x.strip() for x in args.runs.split(",") if x.strip()} or None

    run_dirs = _iter_eval_dirs(eval_root, run_names)
    if not run_dirs:
        print(f"No run folders found in {eval_root}", file=sys.stderr)
        return 1

    if args.artifact:
        os.environ["WANDB_LOG_ARTIFACT"] = "1"

    if args.group:
        os.environ["WANDB_RUN_GROUP"] = args.group

    if not args.dry_run and not (os.environ.get("WANDB_PROJECT") or "").strip():
        print("WANDB_PROJECT is required (or pass --dry-run).", file=sys.stderr)
        return 1

    if not args.dry_run:
        # Imported lazily so dry-run does not require wandb package.
        from wandb_tracking import wandb_eval_finish, wandb_eval_init, wandb_eval_log_row

    uploaded = 0
    skipped = 0

    for run_dir in run_dirs:
        csv_path = _pick_csv(run_dir)
        if csv_path is None:
            skipped += 1
            print(f"skip {run_dir.name}: no ablation_results.csv or assetops_metrics.csv")
            continue

        n_rows = _count_rows(csv_path)
        if n_rows == 0:
            skipped += 1
            print(f"skip {run_dir.name}: {csv_path.name} has 0 rows")
            continue

        if args.dry_run:
            extras = []
            if (run_dir / "assetops_metrics.csv").is_file():
                extras.append("assetops_metrics.csv")
            if (run_dir / "assetops_metrics_by_condition.csv").is_file():
                extras.append("assetops_metrics_by_condition.csv")
            extra_s = f"; tables={','.join(extras)}" if (args.log_tables and extras) else ""
            print(f"would upload {run_dir.name}: {csv_path.name} ({n_rows} rows){extra_s}")
            uploaded += 1
            continue

        os.environ["WANDB_RUN_NAME"] = run_dir.name
        cfg = {
            "source": "backfill",
            "eval_dir": str(run_dir),
            "source_csv": csv_path.name,
            "row_count": n_rows,
        }
        wandb_eval_init(cfg)
        csv.field_size_limit(min(2**31 - 1, 10_000_000))
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                wandb_eval_log_row(row)
        if args.log_tables:
            _log_assetops_tables(run_dir)
        wandb_eval_finish(csv_path if args.artifact else None)
        print(f"uploaded {run_dir.name}: {csv_path.name} ({n_rows} rows)")
        uploaded += 1

    print(f"done: uploaded={uploaded}, skipped={skipped}, root={eval_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

