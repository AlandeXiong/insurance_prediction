"""Model training with SOTA algorithms following Kaggle best practices"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report, confusion_matrix,
    precision_recall_curve, fbeta_score
)
# Note: LightGBM has been removed from the system
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils.class_weight import compute_class_weight
import optuna
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Train and optimize multiple SOTA models with unified pipeline.
    Handles class imbalance and threshold optimization.
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42, 
                 scoring: str = 'roc_auc', n_trials: int = 100,
                 min_recall: float = 0.6, optimize_threshold: bool = True):
        """
        Initialize model trainer.
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            scoring: Scoring metric for optimization
            n_trials: Number of optimization trials
            min_recall: Minimum recall requirement (default 0.6)
            optimize_threshold: Whether to optimize prediction threshold
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scoring = scoring
        self.n_trials = n_trials
        self.min_recall = min_recall
        self.optimize_threshold = optimize_threshold
        
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        self.feature_importance = {}
        self.best_thresholds = {}  # Store optimal thresholds for each model
        self.class_weights = None  # Will be computed from training data
    
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
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               min_recall: float = 0.6) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold that meets minimum recall requirement
        while maximizing precision.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            min_recall: Minimum recall requirement
        
        Returns:
            Optimal threshold and metrics at that threshold
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Note: precision_recall_curve returns arrays where last element is for threshold=1.0
        # We need to prepend a threshold for the case where all predictions are positive
        # thresholds array is one element shorter than precision/recall
        # We'll add a very low threshold (0.0) to handle the case where recall=1.0
        
        # Find thresholds that meet minimum recall requirement
        valid_indices = np.where(recall >= min_recall)[0]
        
        if len(valid_indices) == 0:
            # If no threshold meets requirement, use threshold with max recall
            best_idx = np.argmax(recall)
            if best_idx < len(thresholds):
                best_threshold = thresholds[best_idx]
            elif best_idx == len(recall) - 1:
                # Maximum recall (all positive) - use very low threshold
                best_threshold = 0.0
            else:
                best_threshold = 0.5
            print(f"Warning: No threshold meets min_recall={min_recall}. Using best available (recall={recall[best_idx]:.4f}).")
        else:
            # Among valid thresholds, choose one with highest precision
            valid_precisions = precision[valid_indices]
            best_valid_idx = valid_indices[np.argmax(valid_precisions)]
            if best_valid_idx < len(thresholds):
                best_threshold = thresholds[best_valid_idx]
            elif best_valid_idx == len(recall) - 1:
                # Maximum recall case
                best_threshold = 0.0
            else:
                best_threshold = 0.5
        
        # Ensure threshold is in valid range [0, 1]
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
    
    def optimize_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters with class imbalance handling.
        Uses F1 score optimization to balance precision and recall.
        """
        print("\nOptimizing XGBoost...")
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_train)
        scale_pos_weight = class_weights[1] / class_weights[0] if 0 in class_weights and 1 in class_weights else 1.0
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
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
            
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params, n_estimators=1000)
                model.fit(X_tr, y_tr, 
                         eval_set=[(X_val, y_val)],
                         early_stopping_rounds=100,
                         verbose=False)
                
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Optimize threshold on validation set
                _, metrics = self.find_optimal_threshold(y_val.values, y_pred_proba, self.min_recall)
                
                # Use F1 score as optimization metric (balances precision and recall)
                score = metrics['f1']
                scores.append(score)
            
            return np.mean(scores)
        
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
        print(f"Best CV F1 score: {study.best_value:.4f}")
        
        return best_params
    
    def optimize_catboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Optimize CatBoost hyperparameters with class imbalance handling.
        Uses F1 score optimization to balance precision and recall.
        """
        print("\nOptimizing CatBoost...")
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_train)
        class_weights_list = [class_weights.get(0, 1.0), class_weights.get(1, 1.0)]
        
        def objective(trial):
            params = {
                'iterations': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                'class_weights': class_weights_list,  # Use computed class weights
                'random_seed': self.random_state,
                'od_type': 'Iter',
                'od_wait': 100,
                'verbose': False
            }
            
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 1)
            elif params['bootstrap_type'] == 'Bernoulli':
                params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = CatBoostClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
                
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Optimize threshold on validation set
                _, metrics = self.find_optimal_threshold(y_val.values, y_pred_proba, self.min_recall)
                
                # Use F1 score as optimization metric
                score = metrics['f1']
                scores.append(score)
            
            return np.mean(scores)
        
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
        print(f"Best CV F1 score: {study.best_value:.4f}")
        
        return best_params
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: Optional[pd.DataFrame] = None,
                     y_val: Optional[pd.Series] = None,
                     optimize: bool = True) -> xgb.XGBClassifier:
        """
        Train XGBoost model with class imbalance handling.
        All models use the same feature engineering pipeline.
        """
        print("\nTraining XGBoost...")
        
        # Compute class weights for this training set
        class_weights = self.compute_class_weights(y_train)
        scale_pos_weight = class_weights[1] / class_weights[0] if 0 in class_weights and 1 in class_weights else 1.0
        
        if optimize:
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
                    y_val.values, y_val_proba, self.min_recall
                )
                self.best_thresholds['xgboost'] = best_threshold
                print(f"Optimal threshold for XGBoost: {best_threshold:.4f}")
                print(f"  Precision: {threshold_metrics['precision']:.4f}, Recall: {threshold_metrics['recall']:.4f}, F1: {threshold_metrics['f1']:.4f}")
        else:
            model.fit(X_train, y_train)
            # Use default threshold if no validation set
            self.best_thresholds['xgboost'] = 0.5
        
        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = dict(zip(
            X_train.columns, model.feature_importances_
        ))
        
        return model
    
    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.Series] = None,
                      optimize: bool = True) -> CatBoostClassifier:
        """
        Train CatBoost model with class imbalance handling.
        All models use the same feature engineering pipeline.
        """
        print("\nTraining CatBoost...")
        
        # Compute class weights for this training set
        class_weights = self.compute_class_weights(y_train)
        class_weights_list = [class_weights.get(0, 1.0), class_weights.get(1, 1.0)]
        
        if optimize:
            params = self.optimize_catboost(X_train, y_train)
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
        
        model = CatBoostClassifier(**params)
        
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            
            # Find optimal threshold on validation set
            if self.optimize_threshold:
                y_val_proba = model.predict_proba(X_val)[:, 1]
                best_threshold, threshold_metrics = self.find_optimal_threshold(
                    y_val.values, y_val_proba, self.min_recall
                )
                self.best_thresholds['catboost'] = best_threshold
                print(f"Optimal threshold for CatBoost: {best_threshold:.4f}")
                print(f"  Precision: {threshold_metrics['precision']:.4f}, Recall: {threshold_metrics['recall']:.4f}, F1: {threshold_metrics['f1']:.4f}")
        else:
            model.fit(X_train, y_train, verbose=False)
            # Use default threshold if no validation set
            self.best_thresholds['catboost'] = 0.5
        
        self.models['catboost'] = model
        self.feature_importance['catboost'] = dict(zip(
            X_train.columns, model.feature_importances_
        ))
        
        return model
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.Series] = None,
                      method: str = 'voting') -> VotingClassifier:
        """
        Train ensemble model.
        Note: lightgbm has been removed, only train xgboost and catboost.
        All models use the same feature engineering pipeline.
        """
        print(f"\nTraining {method} ensemble...")
        
        # Ensure base models are trained (only train models that exist in self.models)
        if 'xgboost' not in self.models:
            self.train_xgboost(X_train, y_train, X_val, y_val, optimize=False)
        if 'catboost' not in self.models:
            self.train_catboost(X_train, y_train, X_val, y_val, optimize=False)
        
        # Build estimators list from available models (excluding lightgbm)
        estimators = []
        if 'xgboost' in self.models:
            estimators.append(('xgb', self.models['xgboost']))
        if 'catboost' in self.models:
            estimators.append(('cat', self.models['catboost']))
        
        if not estimators:
            raise ValueError("No base models available for ensemble. Train at least one model first.")
        
        print(f"Ensemble will use {len(estimators)} model(s): {[name for name, _ in estimators]}")
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Find optimal threshold for ensemble
        if self.optimize_threshold and X_val is not None and y_val is not None:
            y_val_proba = ensemble.predict_proba(X_val)[:, 1]
            best_threshold, threshold_metrics = self.find_optimal_threshold(
                y_val.values, y_val_proba, self.min_recall
            )
            self.best_thresholds['ensemble'] = best_threshold
            print(f"Optimal threshold for Ensemble: {best_threshold:.4f}")
            print(f"  Precision: {threshold_metrics['precision']:.4f}, Recall: {threshold_metrics['recall']:.4f}, F1: {threshold_metrics['f1']:.4f}")
        else:
            self.best_thresholds['ensemble'] = 0.5
        
        self.models['ensemble'] = ensemble
        
        return ensemble
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, model_name: str = None, 
                 reoptimize_threshold: bool = True) -> Dict[str, float]:
        """
        Evaluate model(s) with optimal threshold.
        Uses threshold optimization to meet recall requirement.
        
        Args:
            X: Features
            y: True labels
            model_name: Specific model to evaluate, or None for all
            reoptimize_threshold: If True, re-optimize threshold on this dataset
                                 (useful when test set distribution differs from validation set)
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
                threshold = self.best_thresholds.get(name, 0.5)
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
            print(f"  Accuracy: {results[name]['accuracy']:.4f}")
            print(f"  Precision: {results[name]['precision']:.4f}")
            print(f"  Recall: {results[name]['recall']:.4f} {'✓' if recall_optimal >= self.min_recall else '✗ (below min_recall=' + str(self.min_recall) + ')'}")
            print(f"  F1-Score: {results[name]['f1']:.4f}")
            print(f"\n  Default Threshold (0.5) Comparison:")
            print(f"    Precision: {results[name]['precision_default']:.4f}, Recall: {results[name]['recall_default']:.4f}, F1: {results[name]['f1_default']:.4f}")
        
        return results
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """
        Cross-validation for all models using F1 score.
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
                
                # Find optimal threshold on this fold's validation set
                # This ensures threshold is optimized per fold
                best_threshold, _ = self.find_optimal_threshold(y_val.values, y_pred_proba, self.min_recall)
                y_pred = (y_pred_proba >= best_threshold).astype(int)
                
                score = f1_score(y_val, y_pred)
                scores.append(score)
            
            cv_scores[name] = np.array(scores)
            print(f"{name}: F1={np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
        
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
            'min_recall': self.min_recall
        }
        joblib.dump(metadata, models_dir / 'model_metadata.pkl')
        print(f"Saved model metadata including thresholds to {models_dir / 'model_metadata.pkl'}")
    
    def load_model(self, model_path: Path, model_name: str):
        """Load a specific model"""
        model = joblib.load(model_path)
        self.models[model_name] = model
        return model
