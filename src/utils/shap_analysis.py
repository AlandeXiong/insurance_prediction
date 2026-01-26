"""
SHAP analysis utilities (optional).

Designed to be safe-by-default:
- If `shap` isn't installed, functions will raise ImportError (caller should catch and continue).
- Uses sampling to keep runtime bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShapArtifacts:
    output_dir: Path
    beeswarm_png: Optional[Path]
    bar_png: Optional[Path]
    mean_abs_csv: Optional[Path]
    meta_json: Optional[Path]


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sample_df(X: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(X) <= n:
        return X
    return X.sample(n=int(n), random_state=int(seed))


def _mean_abs_values(values: np.ndarray) -> np.ndarray:
    """
    Normalize SHAP values to (n_samples, n_features), then compute mean(|shap|) per feature.
    Handles:
    - (n, f)
    - (n, f, k)  -> uses positive class index 1 if k>=2 else 0
    """
    arr = np.asarray(values)
    if arr.ndim == 2:
        return np.mean(np.abs(arr), axis=0)
    if arr.ndim == 3:
        k = arr.shape[2]
        idx = 1 if k >= 2 else 0
        return np.mean(np.abs(arr[:, :, idx]), axis=0)
    raise ValueError(f"Unsupported SHAP values shape: {arr.shape}")


def run_shap_analysis(
    *,
    model: Any,
    model_name: str,
    X: pd.DataFrame,
    output_dir: Path,
    sample_size: int = 1000,
    background_size: int = 200,
    max_display: int = 20,
    random_state: int = 42,
) -> ShapArtifacts:
    """
    Compute SHAP explanations on a sample of X and save plots + mean(|shap|) CSV.
    """
    # Optional dependency
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import json
    import shap

    _safe_mkdir(output_dir)

    X_sample = _sample_df(X, int(sample_size), int(random_state))
    X_bg = _sample_df(X_sample, int(background_size), int(random_state))

    # Build explainer. For tree models this will pick TreeExplainer automatically.
    explainer = shap.Explainer(model, X_bg)
    shap_values = explainer(X_sample)

    # Save mean(|shap|) per feature for easy consumption
    feature_names: Sequence[str] = list(getattr(shap_values, "feature_names", None) or list(X_sample.columns))
    mean_abs = _mean_abs_values(getattr(shap_values, "values"))
    mean_abs_df = (
        pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    mean_abs_csv = output_dir / "shap_mean_abs.csv"
    mean_abs_df.to_csv(mean_abs_csv, index=False)

    # Plots
    beeswarm_png = output_dir / "shap_beeswarm.png"
    bar_png = output_dir / "shap_bar.png"

    # beeswarm
    try:
        plt.figure()
        shap.plots.beeswarm(shap_values, max_display=int(max_display), show=False)
        plt.title(f"SHAP beeswarm - {model_name}")
        plt.tight_layout()
        plt.savefig(beeswarm_png, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        beeswarm_png = None

    # bar (global importance)
    try:
        plt.figure()
        shap.plots.bar(shap_values, max_display=int(max_display), show=False)
        plt.title(f"SHAP bar - {model_name}")
        plt.tight_layout()
        plt.savefig(bar_png, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        bar_png = None

    meta = {
        "model_name": str(model_name),
        "n_rows_total": int(len(X)),
        "n_rows_sample": int(len(X_sample)),
        "n_rows_background": int(len(X_bg)),
        "max_display": int(max_display),
    }
    meta_json = output_dir / "shap_meta.json"
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return ShapArtifacts(
        output_dir=output_dir,
        beeswarm_png=beeswarm_png,
        bar_png=bar_png,
        mean_abs_csv=mean_abs_csv,
        meta_json=meta_json,
    )

