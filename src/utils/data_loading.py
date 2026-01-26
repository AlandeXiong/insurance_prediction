"""
Unified data loading / splitting utilities.

This module provides ONE authoritative way to obtain (df_train, df_test) based on config.yaml.

Supported strategies (data.strategy):
- pre_split: use already-split CSV files on disk
- timecut: split the original source file by a date cutoff (time-based holdout)
- ratio: random split the original source file by test_size (optionally stratified)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


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
    """
    data_cfg = (config.get("data", {}) or {})
    target_column = str(data_cfg.get("target_column", "Response"))
    resolved = resolve_split(config)

    if resolved.strategy == "pre_split":
        train_path = _resolve_path(resolved.details["train_path"])
        test_path = _resolve_path(resolved.details["test_path"])
        if not train_path.exists():
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        return df_train, df_test, resolved

    if resolved.strategy == "timecut":
        source_path = _resolve_path(resolved.details["source_path"])
        date_col = str(resolved.details["date_col"])
        cutoff = str(resolved.details["cutoff"])
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        df = pd.read_csv(source_path)
        if date_col not in df.columns:
            raise ValueError(f"timecut date_col='{date_col}' not found in source columns")
        dt = pd.to_datetime(df[date_col], errors="coerce")
        if dt.isna().all():
            raise ValueError(f"timecut date_col='{date_col}' could not be parsed as datetime")
        cutoff_dt = pd.to_datetime(cutoff, errors="raise")
        train_mask = dt <= cutoff_dt
        df_train = df.loc[train_mask].copy()
        df_test = df.loc[~train_mask].copy()
        if df_train.empty or df_test.empty:
            raise ValueError(
                f"timecut produced empty split (train={df_train.shape}, test={df_test.shape}); "
                f"check cutoff='{cutoff}' and date_col='{date_col}'"
            )
        return df_train, df_test, resolved

    if resolved.strategy == "ratio":
        source_path = _resolve_path(resolved.details["source_path"])
        test_size = float(resolved.details["test_size"])
        stratify = bool(resolved.details["stratify"])
        random_state = int(resolved.details["random_state"])
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        df = pd.read_csv(source_path)
        stratify_y = df[target_column] if (stratify and target_column in df.columns) else None
        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )
        return df_train.copy(), df_test.copy(), resolved

    # Should be unreachable because resolve_split validates strategy
    raise ValueError(f"Unsupported split strategy: {resolved.strategy}")

