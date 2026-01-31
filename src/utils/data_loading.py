"""
Unified data loading / splitting utilities.

This module provides ONE authoritative way to obtain (df_train, df_test) based on config.yaml.

Supported strategies (data.strategy):
- pre_split: use already-split CSV files on disk
- timecut: split the original source file by a date cutoff (time-based holdout)
- ratio: random split the original source file by test_size (optionally stratified)

Optional: data.deduplicate
- When enabled: load full data -> dedupe by feature columns -> save two files (unique + duplicates),
  then split train/test from the UNIQUE data only.
- Feature columns = config features.categorical_features + numerical_features (present in data).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedSplit:
    strategy: str
    details: Dict[str, Any]


def _repo_root() -> Path:
    # This file lives in src/utils/, so repo root is 3 parents up.
    return Path(__file__).resolve().parents[2]


def _resolve_path(p: str | Path) -> Path:
    pth = Path(p)
    if pth.is_absolute():
        return pth
    return _repo_root() / pth


def _get_dedupe_subset_columns(config: Dict[str, Any], df: pd.DataFrame) -> List[str]:
    """Return feature columns to use for duplicate detection (must exist in df)."""
    feats = (config.get("features") or {}) or {}
    cat = list(feats.get("categorical_features") or [])
    num = list(feats.get("numerical_features") or [])
    drop = set(feats.get("drop_features") or [])
    target = str((config.get("data") or {}).get("target_column", "Response"))
    exclude = drop | {target}
    candidates = [c for c in cat + num if c not in exclude and c in df.columns]
    return candidates


def _load_full_data(config: Dict[str, Any], resolved: ResolvedSplit) -> pd.DataFrame:
    """Load full dataset: for pre_split = train+test concat; for timecut/ratio = source."""
    data_cfg = config.get("data") or {}
    target_column = str(data_cfg.get("target_column", "Response"))

    if resolved.strategy == "pre_split":
        train_path = _resolve_path(resolved.details["train_path"])
        test_path = _resolve_path(resolved.details["test_path"])
        if not train_path.exists():
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        return pd.concat([df_train, df_test], ignore_index=True)

    if resolved.strategy in ("timecut", "ratio"):
        source_path = _resolve_path(resolved.details["source_path"])
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        return pd.read_csv(source_path)

    raise ValueError(f"Unsupported strategy for full load: {resolved.strategy}")


def _dedupe_and_export(
    df_full: pd.DataFrame,
    config: Dict[str, Any],
    export_unique_path: Path,
    export_duplicates_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deduplicate by feature columns: unique = first occurrence per feature set, duplicates = rest.
    Save unique and duplicates to CSV; return (df_unique, df_duplicates).
    """
    subset = _get_dedupe_subset_columns(config, df_full)
    if not subset:
        logger.warning("Deduplicate: no feature columns found in data; skipping dedupe (returning full as unique).")
        export_unique_path.parent.mkdir(parents=True, exist_ok=True)
        df_full.to_csv(export_unique_path, index=False)
        pd.DataFrame().to_csv(export_duplicates_path, index=False)
        return df_full.copy(), pd.DataFrame()

    dup_mask = df_full.duplicated(subset=subset, keep="first")
    df_unique = df_full.loc[~dup_mask].copy()
    df_duplicates = df_full.loc[dup_mask].copy()

    export_unique_path.parent.mkdir(parents=True, exist_ok=True)
    df_unique.to_csv(export_unique_path, index=False)
    df_duplicates.to_csv(export_duplicates_path, index=False)
    logger.info(
        f"Deduplicate by features: unique={len(df_unique)}, duplicates={len(df_duplicates)}; "
        f"saved to {export_unique_path} and {export_duplicates_path}"
    )
    return df_unique, df_duplicates


def resolve_split(config: Dict[str, Any]) -> ResolvedSplit:
    """
    Return the resolved split configuration (strategy + normalized details).
    Prefers config['data']['resolved_split'] produced by load_config().
    """
    data_cfg = (config.get("data", {}) or {})
    resolved = (data_cfg.get("resolved_split") or {})
    strategy = str(resolved.get("strategy") or data_cfg.get("strategy") or "pre_split").lower().strip()

    if strategy == "pre_split":
        ps = data_cfg.get("pre_split", {}) or {}
        train_path = resolved.get("train_path") or ps.get("train_path") or data_cfg.get("train_path")
        test_path = resolved.get("test_path") or ps.get("test_path") or data_cfg.get("test_path")
        if not train_path or not test_path:
            raise ValueError("pre_split requires train_path and test_path")
        return ResolvedSplit(
            strategy="pre_split",
            details={"train_path": str(train_path), "test_path": str(test_path)},
        )

    if strategy == "timecut":
        src = resolved.get("source_path") or data_cfg.get("source_path")
        date_col = resolved.get("date_col") or (data_cfg.get("timecut", {}) or {}).get("date_col", "Effective To Date")
        cutoff = resolved.get("cutoff") or (data_cfg.get("timecut", {}) or {}).get("cutoff")
        if not src or not cutoff:
            raise ValueError("timecut requires source_path and cutoff")
        return ResolvedSplit(
            strategy="timecut",
            details={"source_path": str(src), "date_col": str(date_col), "cutoff": str(cutoff)},
        )

    if strategy == "ratio":
        src = resolved.get("source_path") or data_cfg.get("source_path")
        ra = data_cfg.get("ratio", {}) or {}
        test_size = float(resolved.get("test_size", ra.get("test_size", 0.2)))
        stratify = bool(resolved.get("stratify", ra.get("stratify", True)))
        random_state = int(resolved.get("random_state", ra.get("random_state", data_cfg.get("random_state", 42))))
        if not src:
            raise ValueError("ratio requires source_path")
        return ResolvedSplit(
            strategy="ratio",
            details={
                "source_path": str(src),
                "test_size": float(test_size),
                "stratify": bool(stratify),
                "random_state": int(random_state),
            },
        )

    raise ValueError(f"Unsupported data split strategy: {strategy}")


def load_train_test_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, ResolvedSplit]:
    """
    Load and return (df_train, df_test, resolved_split).

    When data.deduplicate.enabled is True:
    1. Load full data (pre_split: train+test concat; timecut/ratio: source).
    2. Dedupe by feature columns -> df_unique, df_duplicates.
    3. Save unique and duplicates to two CSV files.
    4. Split train/test from df_unique only (by timecut, ratio, or ratio for pre_split).
    """
    data_cfg = (config.get("data", {}) or {})
    target_column = str(data_cfg.get("target_column", "Response"))
    resolved = resolve_split(config)

    # Optional: deduplicate first, then split from unique only
    dedupe_cfg = (data_cfg.get("deduplicate") or {}) or {}
    if bool(dedupe_cfg.get("enabled", False)):
        df_full = _load_full_data(config, resolved)
        export_dir = dedupe_cfg.get("export_dir") or data_cfg.get("export_dir") or "outputs/deduplicated"
        unique_name = str(dedupe_cfg.get("export_unique_name") or "unique.csv")
        duplicates_name = str(dedupe_cfg.get("export_duplicates_name") or "duplicates.csv")
        export_unique_path = _resolve_path(Path(export_dir) / unique_name)
        export_duplicates_path = _resolve_path(Path(export_dir) / duplicates_name)
        df_unique, _ = _dedupe_and_export(
            df_full, config, export_unique_path, export_duplicates_path
        )
        # Split from unique data only (below we use df_unique as the source for each strategy)
        df_to_split = df_unique
    else:
        df_to_split = None

    if resolved.strategy == "pre_split" and df_to_split is None:
        train_path = _resolve_path(resolved.details["train_path"])
        test_path = _resolve_path(resolved.details["test_path"])
        if not train_path.exists():
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        return df_train, df_test, resolved

    if resolved.strategy == "pre_split" and df_to_split is not None:
        # Split unique by ratio (use ratio config)
        ra = data_cfg.get("ratio") or {}
        test_size = float(ra.get("test_size", 0.2))
        stratify = bool(ra.get("stratify", True))
        random_state = int(ra.get("random_state", data_cfg.get("random_state", 42)))
        stratify_y = df_to_split[target_column] if (stratify and target_column in df_to_split.columns) else None
        df_train, df_test = train_test_split(
            df_to_split,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )
        return df_train.copy(), df_test.copy(), resolved

    if resolved.strategy == "timecut":
        if df_to_split is None:
            source_path = _resolve_path(resolved.details["source_path"])
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            df_to_split = pd.read_csv(source_path)
        date_col = str(resolved.details["date_col"])
        cutoff = str(resolved.details["cutoff"])
        if date_col not in df_to_split.columns:
            raise ValueError(f"timecut date_col='{date_col}' not found in source columns")
        dt = pd.to_datetime(df_to_split[date_col], errors="coerce")
        if dt.isna().all():
            raise ValueError(f"timecut date_col='{date_col}' could not be parsed as datetime")
        cutoff_dt = pd.to_datetime(cutoff, errors="raise")
        train_mask = dt <= cutoff_dt
        df_train = df_to_split.loc[train_mask].copy()
        df_test = df_to_split.loc[~train_mask].copy()
        if df_train.empty or df_test.empty:
            raise ValueError(
                f"timecut produced empty split (train={df_train.shape}, test={df_test.shape}); "
                f"check cutoff='{cutoff}' and date_col='{date_col}'"
            )
        return df_train, df_test, resolved

    if resolved.strategy == "ratio":
        if df_to_split is None:
            source_path = _resolve_path(resolved.details["source_path"])
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            df_to_split = pd.read_csv(source_path)
        test_size = float(resolved.details["test_size"])
        stratify = bool(resolved.details["stratify"])
        random_state = int(resolved.details["random_state"])
        stratify_y = df_to_split[target_column] if (stratify and target_column in df_to_split.columns) else None
        df_train, df_test = train_test_split(
            df_to_split,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )
        return df_train.copy(), df_test.copy(), resolved

    raise ValueError(f"Unsupported split strategy: {resolved.strategy}")

