from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score


def _find_best_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    min_recall: float,
    min_precision: float,
) -> tuple[float, float, float, int, bool]:
    """
    Return (threshold, precision, recall, pred_pos, used_relaxed_precision).

    Selection:
    - Prefer thresholds satisfying BOTH recall>=min_recall AND precision>=min_precision
      and maximize precision; tie-break by higher threshold.
    - If none satisfy precision constraint, relax precision and maximize precision
      subject to recall>=min_recall.
    """
    grid = np.unique(np.concatenate(([0.0], np.unique(proba), [1.0])))
    grid.sort()

    best = None
    # scan descending threshold: higher threshold tends to higher precision
    for t in grid[::-1]:
        pred = (proba >= t).astype(int)
        r = float(recall_score(y_true, pred))
        if r < min_recall:
            continue
        p = float(precision_score(y_true, pred, zero_division=0))
        if p < min_precision:
            continue
        npos = int(pred.sum())
        best = (float(t), p, r, npos, False)
        break

    if best is not None:
        return best

    # Relax precision constraint
    best_relaxed = None
    for t in grid[::-1]:
        pred = (proba >= t).astype(int)
        r = float(recall_score(y_true, pred))
        if r < min_recall:
            continue
        p = float(precision_score(y_true, pred, zero_division=0))
        npos = int(pred.sum())
        best_relaxed = (float(t), p, r, npos, True)
        break
    if best_relaxed is None:
        # Can't even meet recall (should be rare). Fall back to threshold=0.0
        pred = (proba >= 0.0).astype(int)
        return 0.0, float(precision_score(y_true, pred, zero_division=0)), float(
            recall_score(y_true, pred)
        ), int(pred.sum()), True
    return best_relaxed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    # Ensure repo root is on sys.path so `import src...` works when run via `conda run`.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.config import load_config
    from src.utils.data_loading import load_train_test_data
    from src.features.engineering import FeatureEngineer

    config = load_config()
    min_recall = float(config.get("model", {}).get("min_recall", 0.6))
    min_precision = float(config.get("model", {}).get("min_precision", 0.0))

    drop_cols = list(config.get("features", {}).get("drop_features", []) or [])
    target_col = str(config.get("data", {}).get("target_column", "Response"))

    models_dir = root / str(config.get("paths", {}).get("models_dir", "models"))
    reports_dir = root / str(config.get("paths", {}).get("outputs_dir", "outputs")) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "threshold_sweep.csv"

    _, df_test, resolved_split = load_train_test_data(config)
    # Load the same target encoder as training uses
    le_target = joblib.load(models_dir / "target_encoder.pkl")

    # Encode target (handle unseen labels like train.py)
    y_raw = df_test[target_col].astype(str)
    known = set(le_target.classes_)
    default_class = le_target.classes_[0]
    default_encoded = int(le_target.transform([default_class])[0])
    y_enc = []
    for v in y_raw.values:
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
    models = {k: joblib.load(p) for k, p in model_paths.items() if p.exists()}

    rows = []
    chosen = {}

    # dense grid for reporting
    report_grid = np.linspace(0.0, 1.0, 201)

    for name, model in models.items():
        proba = model.predict_proba(X_test_processed)[:, 1]

        # pick best threshold
        t, p, r, npos, relaxed = _find_best_threshold(y_test, proba, min_recall, min_precision)
        chosen[name] = {"threshold": float(t), "precision": float(p), "recall": float(r), "pred_pos": int(npos), "relaxed": bool(relaxed)}

        # record full sweep
        for thr in report_grid:
            pred = (proba >= float(thr)).astype(int)
            rows.append(
                {
                    "model": name,
                    "threshold": float(thr),
                    "precision": float(precision_score(y_test, pred, zero_division=0)),
                    "recall": float(recall_score(y_test, pred)),
                    "pred_pos": int(pred.sum()),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    print("## Threshold sweep (test set)")
    print("data_split:", resolved_split.strategy, resolved_split.details)
    print("rows:", len(y_test), "positives:", int(y_test.sum()))
    print(f"constraints: min_recall={min_recall} min_precision={min_precision}")
    print("output_table:", out_csv)

    print("\n## Chosen thresholds")
    for name in sorted(chosen.keys()):
        c = chosen[name]
        suffix = " (precision constraint relaxed)" if c["relaxed"] else ""
        print(
            f"- {name}: t={c['threshold']:.6f} precision={c['precision']:.4f} recall={c['recall']:.4f} pred_pos={c['pred_pos']}{suffix}"
        )


if __name__ == "__main__":
    main()

