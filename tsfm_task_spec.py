"""Parse AssetOpsBench TSFM inference scenarios and resolve dataset CSV paths.

**Hugging Face vs tabular fixtures**

The public dataset `ibm-research/AssetOpsBench` on Hugging Face (see its README / dataset
card) exposes **scenario text** as JSONL configs—for example ``data/scenarios/all_utterance.jsonl``
under the ``scenarios`` config. That gives you **queries** (including TSFM prompts that
*mention* filenames like ``chiller9_annotated_small_test.csv``).

The HF README **does not** ship those CSV time-series files as part of the listed
``data_files``. The paths in prompts are meant to be resolved against a **local**
AssetOpsBench / TSFM runtime layout (often ``data/tsfm_test_data/`` inside the **software**
repository, Docker image, or another distribution channel—not the scenarios-only HF snapshot).

**Local resolution**

Official prompts (see ``AssetOpsBench/.../tsfm_utterance.json``) name files such as
``chiller9_annotated_small_test.csv`` or ``data/tsfm_test_data/chiller9_tsad-small.csv``.
AssetOpsBench resolves relative paths via ``PATH_TO_DATASETS_DIR`` (see
``servers/tsfm/io.py::_get_dataset_path``). Place the benchmark CSVs there, or under
``<AssetOpsBench repo>/data/tsfm_test_data/``, and set ``ASSETOPS`` to the bench
``src`` directory so we can probe those locations.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OfficialTSFMForecastSpec:
    dataset_ref: str
    target_column: str
    timestamp_column: str
    conditional_columns: tuple[str, ...]


def parse_official_tsfm_forecast_task(task: str) -> OfficialTSFMForecastSpec | None:
    """If ``task`` looks like an official TSFM *inference* forecast prompt, return its fields."""
    t = task.strip()
    if not t:
        return None

    m_ds = re.search(r"""in\s+['"](?P<ds>[^'"]+\.csv)['"]""", t, re.I)
    if not m_ds:
        m_ds = re.search(
            r"""data\s+in\s+(?P<ds>[a-zA-Z0-9_./-]+\.csv)(?:\s|$|[,.'"])""",
            t,
            re.I,
        )
    if not m_ds:
        return None
    dataset_ref = m_ds.group("ds").strip()

    m_tgt = re.search(r"(?:Forecast|forecast)\s+['\"](?P<tg>[^'\"]+)['\"]", t)
    if not m_tgt:
        m_tgt = re.search(r"to\s+forecast\s+['\"](?P<tg>[^'\"]+)['\"]", t, re.I)
    if not m_tgt:
        return None
    target_column = m_tgt.group("tg").strip()

    ts_col = "Timestamp"
    m_ts = re.search(
        r"""parameter\s+['"](?P<ts>[^'"]+)['"]\s+as\s+a\s+timestamp""",
        t,
        re.I,
    )
    if m_ts:
        ts_col = m_ts.group("ts").strip()
    else:
        m_ts2 = re.search(
            r"""with\s+['"](?P<ts>[^'"]+)['"]\s+as\s+a\s+timestamp""",
            t,
            re.I,
        )
        if m_ts2:
            ts_col = m_ts2.group("ts").strip()

    cond: tuple[str, ...] = ()
    m_in = re.search(r"""inputs\s+['"](?P<lst>[^'"]+)['"]""", t, re.I)
    if m_in:
        raw = m_in.group("lst")
        cond = tuple(x.strip() for x in raw.split(",") if x.strip())

    return OfficialTSFMForecastSpec(
        dataset_ref=dataset_ref,
        target_column=target_column,
        timestamp_column=ts_col,
        conditional_columns=cond,
    )


def resolve_tsfm_dataset_path(
    dataset_ref: str,
    *,
    assetops_repo_root: Path | None,
) -> Path | None:
    """Resolve ``dataset_ref`` to an existing file, mirroring bench path rules."""
    ref = dataset_ref.strip().strip("'\"")
    if not ref:
        return None
    if os.path.isabs(ref):
        p = Path(ref)
        return p.resolve() if p.is_file() else None

    candidates: list[Path] = []
    dd = os.getenv("PATH_TO_DATASETS_DIR", "").strip()
    if dd:
        candidates.append(Path(dd) / ref)

    if assetops_repo_root is not None:
        repo = assetops_repo_root.resolve()
        candidates.append(repo / ref)
        base = os.path.basename(ref)
        candidates.append(repo / "data" / "tsfm_test_data" / base)
        if not ref.startswith("data/"):
            candidates.append(repo / "data" / "tsfm_test_data" / ref)

    seen: set[str] = set()
    for p in candidates:
        try:
            key = str(p.resolve())
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        try:
            if p.is_file():
                return p.resolve()
        except OSError:
            continue
    return None
