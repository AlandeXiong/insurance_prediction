"""Model training with SOTA algorithms following Kaggle best practices"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report, confusion_matrix,
    precision_recall_curve, fbeta_score, average_precision_score
)
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None
from sklearn.ensemble import VotingClassifier
from sklearn.utils.class_weight import compute_class_weight
import optuna
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


class ModelTrainer:
    """
    Train and optimize multiple SOTA models with unified pipeline.
    Handles class imbalance and threshold optimization.
    """
    
    def __init__(
        self,
        cv_folds: int = 5,
        random_state: int = 42,
        scoring: str = "roc_auc",
        n_trials: int = 100,
        min_recall: float = 0.6,
        min_precision: float = 0.3,
        optimize_threshold: bool = True,
        threshold_selection: str = "max_precision",
        thresholds: Optional[Dict[str, float]] = None,
        default_threshold: float = 0.5,
    ):
        """
        Initialize model trainer.
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            scoring: Scoring metric for optimization
            n_trials: Number of optimization trials
            min_recall: Minimum recall requirement (default 0.6)
            min_precision: Minimum precision requirement (default 0.3)
            optimize_threshold: Whether to optimize prediction threshold
            threshold_selection: Threshold selection strategy under constraints.
                - "max_precision": maximize precision (may push threshold high)
                - "max_recall": maximize recall under precision constraint (recall-first, more robust)
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scoring = scoring
        self.n_trials = n_trials
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.optimize_threshold = optimize_threshold
        self.threshold_selection = threshold_selection
        self.default_threshold = float(default_threshold)
        
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        self.feature_importance = {}
        # Store thresholds per model (prefer config-provided thresholds; training-time auto-search is optional).
        self.best_thresholds: Dict[str, float] = dict(thresholds or {})
        self.class_weights = None  # Will be computed from training data
        # OOF fold artifacts (feature engineer + fold models) for leakage-free bagging inference
        # (kept for compatibility; fast path does not require them).
        self.oof_baggers_raw: Dict[str, List[Dict[str, Any]]] = {}
        self.ensemble_base_models: List[str] = []

    def _score_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Compute a CV/Optuna score from predicted probabilities.

        This is intentionally threshold-free by default (roc_auc / average_precision),
        so that thresholds can be fully controlled via config.yaml.
        """
        metric = str(self.scoring).lower().strip()
        if metric in {"roc_auc", "auc"}:
            return float(roc_auc_score(y_true, y_pred_proba))
        if metric in {"average_precision", "ap", "pr_auc"}:
            return float(average_precision_score(y_true, y_pred_proba))
        # Fallback: ROC-AUC (robust default)
        return float(roc_auc_score(y_true, y_pred_proba))

    def _holdout_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        holdout_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create a single stratified holdout split for fast hyperparameter tuning.
        This avoids KFold loops (much faster) and is intended for FAST mode only.
        """
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=float(holdout_size),
            random_state=self.random_state,
            stratify=y,
        )
        return X_tr, X_val, y_tr, y_val
    
    def compute_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """
        Compute class weights for imbalanced classification.
        
        Args:
            y: Target series
        
        Returns:
            Dictionary mapping class to weight
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y.values)
        class_weights = dict(zip(classes, weights))
        print(f"Computed class weights: {class_weights}")
        return class_weights

    def _lgbm_default_params(self) -> Dict[str, Any]:
        """Default LightGBM parameters (safe baseline on macOS)."""
        return {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }

    def optimize_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        use_holdout: bool = False,
        holdout_size: float = 0.2,
        tuning_n_estimators: int = 800,
        tuning_early_stopping_rounds: int = 30,
    ) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters with class imbalance handling.
        Optimizes a threshold-free metric (self.scoring) on CV folds.
        Thresholds are controlled via config.yaml (per-model thresholds).
        """
        if lgb is None:  # pragma: no cover
            raise ImportError(
                "lightgbm is not installed. Install it (e.g., `pip install lightgbm`) "
                "or remove 'lightgbm' from config.yaml model lists."
            )
        print("\nOptimizing LightGBM...")
        # Debug: basic target distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"[DEBUG][lightgbm] y_train class distribution: {dict(zip(unique, counts))}")

        class_weights = self.compute_class_weights(y_train)
        scale_pos_weight = class_weights.get(1, 1.0) / class_weights.get(0, 1.0) if 0 in class_weights and 1 in class_weights else 1.0

        if use_holdout:
            X_tr, X_val, y_tr, y_val = self._holdout_split(
                X_train, y_train, holdout_size=holdout_size
            )

        def objective(trial):
            params = {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, scale_pos_weight * 2),
                "random_state": self.random_state,
                "n_jobs": -1,
                "verbosity": -1,
            }

            if use_holdout:
                model = lgb.LGBMClassifier(**params, n_estimators=int(tuning_n_estimators))
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(
                            stopping_rounds=int(tuning_early_stopping_rounds), verbose=False
                        )
                    ],
                )
                proba = model.predict_proba(X_val)[:, 1]
                score = float(self._score_from_proba(y_val.values, proba))
                print(
                    f"[DEBUG][optuna][lightgbm][holdout] "
                    f"trial={trial.number} {self.scoring}={score:.6f}"
                )
                if score > 0.98:
                    print(
                        f"[WARN][optuna][lightgbm][holdout] unusually high {self.scoring}={score:.6f}. "
                        "Check for potential leakage or overly-easy split (time/ID features, target-like columns, etc.)."
                    )
                return score

            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = []
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_tr_f, X_val_f = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr_f, y_val_f = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = lgb.LGBMClassifier(**params, n_estimators=1000)
                model.fit(
                    X_tr_f,
                    y_tr_f,
                    eval_set=[(X_val_f, y_val_f)],
                    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
                )
                proba = model.predict_proba(X_val_f)[:, 1]
                fold_score = float(self._score_from_proba(y_val_f.values, proba))
                scores.append(fold_score)
                print(
                    f"[DEBUG][optuna][lightgbm][cv] trial={trial.number} "
                    f"fold={fold} {self.scoring}={fold_score:.6f}"
                )

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            print(
                f"[DEBUG][optuna][lightgbm][cv] trial={trial.number} "
                f"mean_{self.scoring}={mean_score:.6f} (+/- {std_score * 2:.6f}) "
                f"scores={np.round(scores, 6).tolist()}"
            )
            if mean_score > 0.98:
                print(
                    f"[WARN][optuna][lightgbm][cv] unusually high mean {self.scoring}={mean_score:.6f}. "
                    "This can indicate data leakage or a very easy split. "
                    "Consider running leakage diagnostics (src/leakage/* scripts) and inspecting features."
                )
            return mean_score

        study = optuna.create_study(direction="maximize", study_name="lightgbm_opt")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        best_params = dict(study.best_params)
        best_params.update(
            {
                "objective": "binary",
                "metric": "auc",
                "random_state": self.random_state,
                "n_jobs": -1,
                "verbosity": -1,
            }
        )

        print(f"Best LightGBM params: {best_params}")
        print(f"Best CV {self.scoring}: {study.best_value:.4f}")
        return best_params

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        optimize: bool = True,
        params_override: Optional[Dict[str, Any]] = None,
    ) -> lgb.LGBMClassifier:
        """
        Train LightGBM model with class imbalance handling.
        Uses the same feature pipeline as other models (expects engineered numeric matrix).
        """
        if lgb is None:  # pragma: no cover
            raise ImportError(
                "lightgbm is not installed. Install it (e.g., `pip install lightgbm`) "
                "or remove 'lightgbm' from config.yaml model lists."
            )
        print("\nTraining LightGBM...")

        class_weights = self.compute_class_weights(y_train)
        scale_pos_weight = class_weights.get(1, 1.0) / class_weights.get(0, 1.0) if 0 in class_weights and 1 in class_weights else 1.0

        if params_override is not None:
            params = dict(params_override)
        elif optimize:
            params = self.optimize_lightgbm(X_train, y_train)
            self.best_params["lightgbm"] = params
        else:
            params = self._lgbm_default_params()
            params["scale_pos_weight"] = scale_pos_weight

        # Ensure key params exist
        params.setdefault("objective", "binary")
        params.setdefault("metric", "auc")
        params.setdefault("random_state", self.random_state)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbosity", -1)
        params.setdefault("scale_pos_weight", scale_pos_weight)

        model = lgb.LGBMClassifier(**params, n_estimators=5000)
        if X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
            )

            if self.optimize_threshold:
                y_val_proba = model.predict_proba(X_val)[:, 1]
                best_threshold, threshold_metrics = self.find_optimal_threshold(
                    y_val.values,
                    y_val_proba,
                    min_recall=self.min_recall,
                    min_precision=self.min_precision,
                )
                self.best_thresholds["lightgbm"] = best_threshold
                print(f"Optimal threshold for LightGBM: {best_threshold:.4f}")
                print(
                    f"  Precision: {threshold_metrics['precision']:.4f}, "
                    f"Recall: {threshold_metrics['recall']:.4f}, F1: {threshold_metrics['f1']:.4f}"
                )
        else:
            model.fit(X_train, y_train)
            self.best_thresholds.setdefault("lightgbm", self.default_threshold)

        self.models["lightgbm"] = model
        # Feature importance (gain-based)
        try:
            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                self.feature_importance["lightgbm"] = dict(zip(X_train.columns, importances))
        except Exception:
            self.feature_importance["lightgbm"] = {}

        return model

    @staticmethod
    def _cat_feature_indices(X: pd.DataFrame, cat_feature_names: Optional[List[str]]) -> Optional[List[int]]:
        """
        Convert categorical feature names to CatBoost column indices.
        Works even if categorical columns are label-encoded as integers, as long as
        we pass the indices via `cat_features`.
        """
        if not cat_feature_names:
            return None
        indices = [X.columns.get_loc(c) for c in cat_feature_names if c in X.columns]
        return indices if indices else None


    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               min_recall: float = 0.6, min_precision: float = 0.0) -> Tuple[float, Dict[str, float]]:
        """
        Find an operating threshold that satisfies constraints and optimizes the business goal.

        Primary objective:
        - select a threshold under constraints (recall >= min_recall AND precision >= min_precision)
        - then apply a selection policy:
            * max_precision (default): maximize precision (can produce very high thresholds)
            * max_recall: maximize recall (more conservative / recall-first)
            * max_f1: maximize F1 score (within constraints)

        Notes on sklearn's `precision_recall_curve` output:
        - precision/recall arrays have length = len(thresholds) + 1
        - for each thresholds[i], the corresponding precision/recall are precision[i+1], recall[i+1]
        - precision[0], recall[0] correspond to a threshold below the minimum score (predict all positive)
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            min_recall: Minimum recall requirement
        
        Returns:
            Optimal threshold and metrics at that threshold
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Build aligned candidate points (threshold, precision, recall)
        # Candidate 0: predict all positive (threshold=0.0)
        candidates_t = [0.0]
        candidates_p = [float(precision[0])]
        candidates_r = [float(recall[0])]

        # Candidates for each returned threshold
        for i, t in enumerate(thresholds):
            candidates_t.append(float(t))
            candidates_p.append(float(precision[i + 1]))
            candidates_r.append(float(recall[i + 1]))

        candidates_t = np.array(candidates_t, dtype=float)
        candidates_p = np.array(candidates_p, dtype=float)
        candidates_r = np.array(candidates_r, dtype=float)
        
        # Feasible set: satisfy BOTH constraints
        feasible = np.where((candidates_r >= min_recall) & (candidates_p >= min_precision))[0]
        if feasible.size > 0:
            if self.threshold_selection == "max_recall":
                # Recall-first + robustness against score calibration shift:
                # pick the LOWEST feasible threshold (typically yields the highest/most stable recall on unseen data)
                # while still satisfying both constraints.
                best_idx = feasible[int(np.argmin(candidates_t[feasible]))]
            elif self.threshold_selection == "max_f1":
                # Maximize F1 under constraints; tie-break by higher precision, then lower threshold.
                f1 = (2.0 * candidates_p * candidates_r) / (candidates_p + candidates_r + 1e-12)
                best_f1 = np.max(f1[feasible])
                best_pool = feasible[np.where(np.isclose(f1[feasible], best_f1, rtol=0, atol=1e-12))[0]]
                if best_pool.size > 1:
                    # Higher precision wins; if still tied, prefer lower threshold for robustness.
                    best_p = np.max(candidates_p[best_pool])
                    best_pool = best_pool[np.where(np.isclose(candidates_p[best_pool], best_p, rtol=0, atol=1e-12))[0]]
                    best_idx = int(best_pool[np.argmin(candidates_t[best_pool])])
                else:
                    best_idx = int(best_pool[0])
            else:
                # Max precision, tie-break by higher recall (precision-first)
                best_idx = feasible[np.lexsort((candidates_r[feasible], candidates_p[feasible]))][-1]
        else:
            # Fallbacks (still deterministic and recall-friendly)
            feasible_recall = np.where(candidates_r >= min_recall)[0]
            feasible_precision = np.where(candidates_p >= min_precision)[0]

            if feasible_recall.size > 0:
                # Under recall constraint, choose by strategy
                if self.threshold_selection == "max_recall":
                    # Lowest threshold among those that meet recall (maximize recall stability)
                    best_idx = feasible_recall[int(np.argmin(candidates_t[feasible_recall]))]
                elif self.threshold_selection == "max_f1":
                    # If we can't meet both constraints, maximize F1 among points meeting recall.
                    f1 = (2.0 * candidates_p * candidates_r) / (candidates_p + candidates_r + 1e-12)
                    best_idx = int(feasible_recall[np.argmax(f1[feasible_recall])])
                else:
                    best_idx = feasible_recall[np.argmax(candidates_p[feasible_recall])]
            elif feasible_precision.size > 0:
                # Max recall under precision constraint
                if self.threshold_selection == "max_recall":
                    best_idx = feasible_precision[int(np.argmin(candidates_t[feasible_precision]))]
                elif self.threshold_selection == "max_f1":
                    # Maximize F1 among points meeting precision.
                    f1 = (2.0 * candidates_p * candidates_r) / (candidates_p + candidates_r + 1e-12)
                    best_idx = int(feasible_precision[np.argmax(f1[feasible_precision])])
                else:
                    best_idx = feasible_precision[np.argmax(candidates_r[feasible_precision])]
            else:
                # Nothing meets constraints: maximize F2 (recall-weighted) as best-effort
                beta2 = 2.0
                f2 = (1 + beta2**2) * (candidates_p * candidates_r) / (beta2**2 * candidates_p + candidates_r + 1e-12)
                best_idx = int(np.argmax(f2))
        
        best_threshold = float(candidates_t[best_idx])
        best_threshold = max(0.0, min(1.0, best_threshold))
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
        metrics = {
            'threshold': float(best_threshold),
            'precision': float(precision_score(y_true, y_pred_optimal, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred_optimal)),
            'f1': float(f1_score(y_true, y_pred_optimal)),
            'f2': float(fbeta_score(y_true, y_pred_optimal, beta=2))  # F2 score (recall-weighted)
        }
        
        return best_threshold, metrics
    
    def optimize_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        use_holdout: bool = False,
        holdout_size: float = 0.2,
        tuning_n_estimators: int = 600,
        tuning_early_stopping_rounds: int = 30,
    ) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters with class imbalance handling.
        Optimizes a threshold-free metric (self.scoring) on CV folds.
        Thresholds are controlled via config.yaml (per-model thresholds).
        """
        print("\nOptimizing XGBoost...")
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"[DEBUG][xgboost] y_train class distribution: {dict(zip(unique, counts))}")
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_train)
        scale_pos_weight = class_weights[1] / class_weights[0] if 0 in class_weights and 1 in class_weights else 1.0
        
        if use_holdout:
            X_tr, X_val, y_tr, y_val = self._holdout_split(
                X_train, y_train, holdout_size=holdout_size
            )

        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, scale_pos_weight * 2),
                'random_state': self.random_state,
                'verbosity': 0
            }

            if use_holdout:
                model = xgb.XGBClassifier(**params, n_estimators=int(tuning_n_estimators))
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=int(tuning_early_stopping_rounds),
                    verbose=False,
                )
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = float(self._score_from_proba(y_val.values, y_pred_proba))
                print(
                    f"[DEBUG][optuna][xgboost][holdout] "
                    f"trial={trial.number} {self.scoring}={score:.6f}"
                )
                if score > 0.98:
                    print(
                        f"[WARN][optuna][xgboost][holdout] unusually high {self.scoring}={score:.6f}. "
                        "Check for potential leakage or overly-easy split."
                    )
                return score

            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_tr_f, X_val_f = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr_f, y_val_f = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = xgb.XGBClassifier(**params, n_estimators=1000)
                model.fit(
                    X_tr_f,
                    y_tr_f,
                    eval_set=[(X_val_f, y_val_f)],
                    early_stopping_rounds=100,
                    verbose=False,
                )

                y_pred_proba = model.predict_proba(X_val_f)[:, 1]
                fold_score = float(self._score_from_proba(y_val_f.values, y_pred_proba))
                scores.append(fold_score)
                print(
                    f"[DEBUG][optuna][xgboost][cv] trial={trial.number} "
                    f"fold={fold} {self.scoring}={fold_score:.6f}"
                )

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            print(
                f"[DEBUG][optuna][xgboost][cv] trial={trial.number} "
                f"mean_{self.scoring}={mean_score:.6f} (+/- {std_score * 2:.6f}) "
                f"scores={np.round(scores, 6).tolist()}"
            )
            if mean_score > 0.98:
                print(
                    f"[WARN][optuna][xgboost][cv] unusually high mean {self.scoring}={mean_score:.6f}. "
                    "This can indicate data leakage or a very easy split."
                )
            return mean_score
        
        study = optuna.create_study(direction='maximize', study_name='xgboost_opt')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'verbosity': 0
        })
        
        print(f"Best XGBoost params: {best_params}")
        print(f"Best CV {self.scoring}: {study.best_value:.4f}")
        
        return best_params
    
    def optimize_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cat_feature_names: Optional[List[str]] = None,
        *,
        use_holdout: bool = False,
        holdout_size: float = 0.2,
        tuning_iterations: int = 600,
        tuning_od_wait: int = 30,
    ) -> Dict[str, Any]:
        """
        Optimize CatBoost hyperparameters with class imbalance handling.
        Optimizes a threshold-free metric (self.scoring) on CV folds.
        Thresholds are controlled via config.yaml (per-model thresholds).
        """
        print("\nOptimizing CatBoost...")
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"[DEBUG][catboost] y_train class distribution: {dict(zip(unique, counts))}")
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_train)
        class_weights_list = [class_weights.get(0, 1.0), class_weights.get(1, 1.0)]
        
        if use_holdout:
            X_tr, X_val, y_tr, y_val = self._holdout_split(
                X_train, y_train, holdout_size=holdout_size
            )
            cat_features_holdout = self._cat_feature_indices(X_tr, cat_feature_names)

        def objective(trial):
            params = {
                'iterations': int(tuning_iterations) if use_holdout else 500,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 4, 6),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 5),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                'class_weights': class_weights_list,  # Use computed class weights
                'random_seed': self.random_state,
                'od_type': 'Iter',
                'od_wait': int(tuning_od_wait) if use_holdout else 50,
                'verbose': False
            }
            
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 1)
            elif params['bootstrap_type'] == 'Bernoulli':
                params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            
            if use_holdout:
                model = CatBoostClassifier(**params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_val, y_val),
                    cat_features=cat_features_holdout,
                    verbose=False,
                )
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = float(self._score_from_proba(y_val.values, y_pred_proba))
                print(
                    f"[DEBUG][optuna][catboost][holdout] "
                    f"trial={trial.number} {self.scoring}={score:.6f}"
                )
                if score > 0.98:
                    print(
                        f"[WARN][optuna][catboost][holdout] unusually high {self.scoring}={score:.6f}. "
                        "Check for potential leakage or overly-easy split."
                    )
                return score

            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_tr_f, X_val_f = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr_f, y_val_f = y_train.iloc[train_idx], y_train.iloc[val_idx]
                cat_features = self._cat_feature_indices(X_tr_f, cat_feature_names)

                model = CatBoostClassifier(**params)
                model.fit(
                    X_tr_f,
                    y_tr_f,
                    eval_set=(X_val_f, y_val_f),
                    cat_features=cat_features,
                    verbose=False,
                )

                y_pred_proba = model.predict_proba(X_val_f)[:, 1]
                fold_score = float(self._score_from_proba(y_val_f.values, y_pred_proba))
                scores.append(fold_score)
                print(
                    f"[DEBUG][optuna][catboost][cv] trial={trial.number} "
                    f"fold={fold} {self.scoring}={fold_score:.6f}"
                )

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            print(
                f"[DEBUG][optuna][catboost][cv] trial={trial.number} "
                f"mean_{self.scoring}={mean_score:.6f} (+/- {std_score * 2:.6f}) "
                f"scores={np.round(scores, 6).tolist()}"
            )
            if mean_score > 0.98:
                print(
                    f"[WARN][optuna][catboost][cv] unusually high mean {self.scoring}={mean_score:.6f}. "
                    "This can indicate data leakage or a very easy split."
                )
            return mean_score
        
        study = optuna.create_study(direction='maximize', study_name='catboost_opt')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params.update({
            'iterations': 1000,
            'class_weights': class_weights_list,
            'random_seed': self.random_state,
            'od_type': 'Iter',
            'od_wait': 100,
            'verbose': False
        })
        
        print(f"Best CatBoost params: {best_params}")
        print(f"Best CV {self.scoring}: {study.best_value:.4f}")
        
        return best_params


    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: Optional[pd.DataFrame] = None,
                     y_val: Optional[pd.Series] = None,
                     optimize: bool = True,
                     params_override: Optional[Dict[str, Any]] = None) -> xgb.XGBClassifier:
        """
        Train XGBoost model with class imbalance handling.
        All models use the same feature engineering pipeline.
        """
        print("\nTraining XGBoost...")
        
        # Compute class weights for this training set
        class_weights = self.compute_class_weights(y_train)
        scale_pos_weight = class_weights[1] / class_weights[0] if 0 in class_weights and 1 in class_weights else 1.0
        
        if params_override is not None:
            params = dict(params_override)
        elif optimize:
            params = self.optimize_xgboost(X_train, y_train)
            self.best_params['xgboost'] = params
        else:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
                'random_state': self.random_state,
                'verbosity': 0
            }

        # Ensure key params exist
        params.setdefault('objective', 'binary:logistic')
        params.setdefault('eval_metric', 'auc')
        params.setdefault('random_state', self.random_state)
        params.setdefault('verbosity', 0)
        params.setdefault('scale_pos_weight', scale_pos_weight)
        
        model = xgb.XGBClassifier(**params, n_estimators=1000)
        
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=100,
                     verbose=False)
            
            # Find optimal threshold on validation set
            if self.optimize_threshold:
                y_val_proba = model.predict_proba(X_val)[:, 1]
                best_threshold, threshold_metrics = self.find_optimal_threshold(
                    y_val.values, y_val_proba,
                    min_recall=self.min_recall,
                    min_precision=self.min_precision
                )
                self.best_thresholds['xgboost'] = best_threshold
                print(f"Optimal threshold for XGBoost: {best_threshold:.4f}")
                print(f"  Precision: {threshold_metrics['precision']:.4f}, Recall: {threshold_metrics['recall']:.4f}, F1: {threshold_metrics['f1']:.4f}")
        else:
            model.fit(X_train, y_train)
            # Use default threshold if no validation set
            self.best_thresholds.setdefault('xgboost', self.default_threshold)
        
        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = dict(zip(
            X_train.columns, model.feature_importances_
        ))
        
        return model
    
    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.Series] = None,
                      optimize: bool = True,
                      cat_feature_names: Optional[List[str]] = None,
                      params_override: Optional[Dict[str, Any]] = None) -> CatBoostClassifier:
        """
        Train CatBoost model with class imbalance handling.
        All models use the same feature engineering pipeline.
        """
        print("\nTraining CatBoost...")
        
        # Compute class weights for this training set
        class_weights = self.compute_class_weights(y_train)
        class_weights_list = [class_weights.get(0, 1.0), class_weights.get(1, 1.0)]
        
        if params_override is not None:
            params = dict(params_override)
        elif optimize:
            params = self.optimize_catboost(X_train, y_train, cat_feature_names=cat_feature_names)
            self.best_params['catboost'] = params
        else:
            params = {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'class_weights': class_weights_list,  # Handle class imbalance
                'random_seed': self.random_state,
                'verbose': False
            }

        # Ensure key params exist
        params.setdefault('iterations', 1000)
        params.setdefault('class_weights', class_weights_list)
        params.setdefault('random_seed', self.random_state)
        params.setdefault('verbose', False)
        
        model = CatBoostClassifier(**params)
        cat_features = self._cat_feature_indices(X_train, cat_feature_names)
        
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, verbose=False)
            
            # Find optimal threshold on validation set
            if self.optimize_threshold:
                y_val_proba = model.predict_proba(X_val)[:, 1]
                best_threshold, threshold_metrics = self.find_optimal_threshold(
                    y_val.values, y_val_proba,
                    min_recall=self.min_recall,
                    min_precision=self.min_precision
                )
                self.best_thresholds['catboost'] = best_threshold
                print(f"Optimal threshold for CatBoost: {best_threshold:.4f}")
                print(f"  Precision: {threshold_metrics['precision']:.4f}, Recall: {threshold_metrics['recall']:.4f}, F1: {threshold_metrics['f1']:.4f}")
        else:
            model.fit(X_train, y_train, cat_features=cat_features, verbose=False)
            # Use default threshold if no validation set
            self.best_thresholds.setdefault('catboost', self.default_threshold)
        
        self.models['catboost'] = model
        self.feature_importance['catboost'] = dict(zip(
            X_train.columns, model.feature_importances_
        ))
        
        return model
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.Series] = None,
                      method: str = 'voting',
                      base_models: Optional[List[str]] = None) -> VotingClassifier:
        """
        Train ensemble model.
        All models use the same feature engineering pipeline.
        """
        print(f"\nTraining {method} ensemble...")

        if base_models is None:
            base_models = [m for m in ["lightgbm", "xgboost", "catboost"] if m in self.models]

        # Build estimators list from selected base models (must already be trained for consistency)
        estimators = []
        if "lightgbm" in base_models and "lightgbm" in self.models:
            estimators.append(("lgb", self.models["lightgbm"]))
        if "xgboost" in base_models and "xgboost" in self.models:
            estimators.append(("xgb", self.models["xgboost"]))
        if "catboost" in base_models and "catboost" in self.models:
            estimators.append(("cat", self.models["catboost"]))
        
        if not estimators:
            raise ValueError("No base models available for ensemble. Train base models first.")
        
        print(f"Ensemble will use {len(estimators)} model(s): {[name for name, _ in estimators]}")
        self.ensemble_base_models = [name for name, _ in estimators]
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Find optimal threshold for ensemble
        if self.optimize_threshold and X_val is not None and y_val is not None:
            y_val_proba = ensemble.predict_proba(X_val)[:, 1]
            best_threshold, threshold_metrics = self.find_optimal_threshold(
                y_val.values, y_val_proba,
                min_recall=self.min_recall,
                min_precision=self.min_precision
            )
            self.best_thresholds['ensemble'] = best_threshold
            print(f"Optimal threshold for Ensemble: {best_threshold:.4f}")
            print(f"  Precision: {threshold_metrics['precision']:.4f}, Recall: {threshold_metrics['recall']:.4f}, F1: {threshold_metrics['f1']:.4f}")
        else:
            self.best_thresholds.setdefault('ensemble', self.default_threshold)
        
        self.models['ensemble'] = ensemble
        
        return ensemble
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, model_name: str = None, 
                 reoptimize_threshold: bool = False) -> Dict[str, float]:
        """
        Evaluate model(s) with optimal threshold.
        Uses threshold optimization to meet recall requirement.
        
        Args:
            X: Features
            y: True labels
            model_name: Specific model to evaluate, or None for all
            reoptimize_threshold: If True, re-optimize threshold on this dataset.
                                 WARNING: Do NOT set this to True for the TEST set (label leakage).
        """
        results = {}
        
        models_to_eval = {model_name: self.models[model_name]} if model_name else self.models
        
        # Print class distribution for debugging
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"\nClass distribution: {class_dist}")
        if len(class_dist) == 2:
            pos_ratio = class_dist.get(1, 0) / len(y)
            print(f"Positive class ratio: {pos_ratio:.4f}")
        
        for name, model in models_to_eval.items():
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Re-optimize threshold on this dataset if requested
            if reoptimize_threshold and self.optimize_threshold:
                best_threshold, threshold_metrics = self.find_optimal_threshold(
                    y.values, y_pred_proba, self.min_recall
                )
                print(f"\n{name.upper()} - Threshold optimized on evaluation set:")
                print(f"  Optimal threshold: {best_threshold:.4f}")
                print(f"  Precision: {threshold_metrics['precision']:.4f}")
                print(f"  Recall: {threshold_metrics['recall']:.4f}")
                print(f"  F1: {threshold_metrics['f1']:.4f}")
                threshold = best_threshold
            else:
                # Use stored threshold from validation set
                threshold = self.best_thresholds.get(name, self.default_threshold)
                print(f"\n{name.upper()} - Using stored threshold: {threshold:.4f}")
            
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Also calculate metrics at default threshold for comparison
            y_pred_default = model.predict(X)
            
            # Calculate metrics
            recall_optimal = recall_score(y, y_pred)
            precision_optimal = precision_score(y, y_pred, zero_division=0)
            f1_optimal = f1_score(y, y_pred)
            
            results[name] = {
                'roc_auc': roc_auc_score(y, y_pred_proba),
                'pr_auc': average_precision_score(y, y_pred_proba),
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_optimal,
                'recall': recall_optimal,
                'f1': f1_optimal,
                'threshold': threshold,
                'accuracy_default': accuracy_score(y, y_pred_default),
                'precision_default': precision_score(y, y_pred_default, zero_division=0),
                'recall_default': recall_score(y, y_pred_default),
                'f1_default': f1_score(y, y_pred_default)
            }
            
            print(f"\n{name.upper()} Results (Threshold={threshold:.4f}):")
            print(f"  ROC-AUC: {results[name]['roc_auc']:.4f}")
            print(f"  PR-AUC: {results[name]['pr_auc']:.4f}")
            print(f"  Accuracy: {results[name]['accuracy']:.4f}")
            print(f"  Precision: {results[name]['precision']:.4f}")
            recall_ok = recall_optimal >= self.min_recall
            precision_ok = precision_optimal >= self.min_precision
            print(f"  Recall: {results[name]['recall']:.4f} {'✓' if recall_ok else '✗ (min_recall=' + str(self.min_recall) + ')'}")
            print(f"  Constraint check: Precision>= {self.min_precision} {'✓' if precision_ok else '✗'} | Recall>= {self.min_recall} {'✓' if recall_ok else '✗'}")
            print(f"  F1-Score: {results[name]['f1']:.4f}")
            print(f"\n  Default Threshold (0.5) Comparison:")
            print(f"    Precision: {results[name]['precision_default']:.4f}, Recall: {results[name]['recall_default']:.4f}, F1: {results[name]['f1_default']:.4f}")
        
        return results
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """
        Cross-validation for all models using a threshold-free metric (self.scoring).
        Handles ensemble models (VotingClassifier) specially.
        """
        print("\nPerforming cross-validation...")
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = {}
        
        for name, model in self.models.items():
            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Handle ensemble models (VotingClassifier) specially
                if isinstance(model, VotingClassifier):
                    # For ensemble, retrain base models and recreate ensemble
                    estimators = []
                    for est_name, est_model in model.named_estimators_.items():
                        # Get base model type and params
                        est_type = type(est_model)
                        est_params = est_model.get_params()
                        
                        # Create new instance
                        est_copy = est_type(**est_params)
                        est_copy.fit(X_tr, y_tr)
                        estimators.append((est_name, est_copy))
                    
                    # Create new VotingClassifier with retrained estimators
                    model_copy = VotingClassifier(estimators=estimators, voting='soft')
                    model_copy.fit(X_tr, y_tr)
                else:
                    # For regular models, clone and retrain
                    from sklearn.base import clone
                    model_copy = clone(model)
                    model_copy.fit(X_tr, y_tr)
                
                y_pred_proba = model_copy.predict_proba(X_val)[:, 1]
                scores.append(self._score_from_proba(y_val.values, y_pred_proba))
            
            cv_scores[name] = np.array(scores)
            print(f"{name}: {self.scoring}={np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
        
        self.cv_scores = cv_scores
        return cv_scores
    
    def save_models(self, models_dir: Path):
        """Save all trained models and thresholds"""
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = models_dir / f"{name}_model.pkl"
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save metadata including thresholds
        metadata = {
            'best_params': self.best_params,
            'cv_scores': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                         for k, v in self.cv_scores.items()},
            'feature_importance': self.feature_importance,
            'best_thresholds': self.best_thresholds,
            'min_recall': self.min_recall,
            'min_precision': self.min_precision
        }
        joblib.dump(metadata, models_dir / 'model_metadata.pkl')
        print(f"Saved model metadata including thresholds to {models_dir / 'model_metadata.pkl'}")
    
    def load_model(self, model_path: Path, model_name: str):
        """Load a specific model"""
        model = joblib.load(model_path)
        self.models[model_name] = model
        return model
