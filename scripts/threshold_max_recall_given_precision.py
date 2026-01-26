from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def best_recall_with_precision_constraint(
    y_true: np.ndarray, proba: np.ndarray, min_precision: float
) -> tuple[float, float, float, int] | None:
    """
    Find threshold that maximizes recall subject to precision >= min_precision.
    Returns (threshold, precision, recall, pred_pos). If infeasible, return None.

    Implementation: brute-force over unique predicted probabilities.
    """
    grid = np.unique(np.concatenate(([0.0], np.unique(proba), [1.0])))
    grid.sort()

    best = None
    for t in grid:
        pred = (proba >= float(t)).astype(int)
        p = float(precision_score(y_true, pred, zero_division=0))
        if p < min_precision:
            continue
        r = float(recall_score(y_true, pred))
        npos = int(pred.sum())
        # maximize recall; tie-break higher precision; then higher threshold (more conservative)
        key = (r, p, float(t))
        if best is None or key > best[0]:
            best = (key, float(t), p, r, npos)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.utils.config import load_config
    from src.utils.data_loading import load_train_test_data
    from src.features.engineering import FeatureEngineer

    config = load_config()
    min_precision = 0.30

    target_col = str(config.get("data", {}).get("target_column", "Response"))
    drop_cols = list(config.get("features", {}).get("drop_features", []) or [])

    models_dir = root / str(config.get("paths", {}).get("models_dir", "models"))
    reports_dir = root / str(config.get("paths", {}).get("outputs_dir", "outputs")) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "threshold_max_recall_precision_ge_0p30.csv"

    _, df_test, resolved_split = load_train_test_data(config)

    # encode y using saved encoder
    le_target = joblib.load(models_dir / "target_encoder.pkl")
    known = set(le_target.classes_)
    default_class = le_target.classes_[0]
    default_encoded = int(le_target.transform([default_class])[0])
    y_enc = []
    for v in df_test[target_col].astype(str).values:
        if v in known:
            y_enc.append(int(le_target.transform([v])[0]))
        else:
            y_enc.append(default_encoded)
    y_test = np.array(y_enc, dtype=int)

    df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])
    X_test = df_test.drop(columns=[target_col])

    fe = FeatureEngineer.load(models_dir / "feature_engineer.pkl")
    X_test_processed = fe.transform(X_test, fit=False)

    model_paths = {
        "lightgbm": models_dir / "lightgbm_model.pkl",
        "xgboost": models_dir / "xgboost_model.pkl",
        "catboost": models_dir / "catboost_model.pkl",
        "ensemble": models_dir / "ensemble_model.pkl",
    }

    rows = []
    print("## Max recall with precision >= 0.30 (test set)")
    print("data_split:", resolved_split.strategy, resolved_split.details)
    print("rows:", len(y_test), "positives:", int(y_test.sum()))

    for name, pth in model_paths.items():
        if not pth.exists():
            continue
        model = joblib.load(pth)
        proba = model.predict_proba(X_test_processed)[:, 1]
        best = best_recall_with_precision_constraint(y_test, proba, min_precision=min_precision)
        if best is None:
            print(f"- {name}: infeasible (precision>=0.30 cannot be met at any threshold)")
            continue
        t, p, r, npos = best
        rows.append({"model": name, "threshold": t, "precision": p, "recall": r, "pred_pos": npos})
        print(f"- {name}: recall={r:.4f} at t={t:.6f} (precision={p:.4f}, pred_pos={npos})")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("\nWrote:", out_csv)


if __name__ == "__main__":
    main()

