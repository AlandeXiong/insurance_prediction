"""Model training with SOTA algorithms"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report, confusion_matrix
)
# Note: LightGBM has been removed from the system
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import optuna
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and optimize multiple SOTA models"""
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42, 
                 scoring: str = 'roc_auc', n_trials: int = 100):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scoring = scoring
        self.n_trials = n_trials
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        self.feature_importance = {}
        
    # LightGBM methods have been removed from the system
    
    def optimize_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters"""
        print("\nOptimizing XGBoost...")
        
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
                score = roc_auc_score(y_val, y_pred_proba)
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
        print(f"Best CV score: {study.best_value:.4f}")
        
        return best_params
    
    def optimize_catboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Optimize CatBoost hyperparameters"""
        print("\nOptimizing CatBoost...")
        
        def objective(trial):
            params = {
                'iterations': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
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
                score = roc_auc_score(y_val, y_pred_proba)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', study_name='catboost_opt')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params.update({
            'iterations': 1000,
            'random_seed': self.random_state,
            'od_type': 'Iter',
            'od_wait': 100,
            'verbose': False
        })
        
        print(f"Best CatBoost params: {best_params}")
        print(f"Best CV score: {study.best_value:.4f}")
        
        return best_params
    
    # LightGBM training method has been removed from the system
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: Optional[pd.DataFrame] = None,
                     y_val: Optional[pd.Series] = None,
                     optimize: bool = True) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        print("\nTraining XGBoost...")
        
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
                'random_state': self.random_state,
                'verbosity': 0
            }
        
        model = xgb.XGBClassifier(**params, n_estimators=1000)
        
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=100,
                     verbose=False)
        else:
            model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = dict(zip(
            X_train.columns, model.feature_importances_
        ))
        
        return model
    
    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.Series] = None,
                      optimize: bool = True) -> CatBoostClassifier:
        """Train CatBoost model"""
        print("\nTraining CatBoost...")
        
        if optimize:
            params = self.optimize_catboost(X_train, y_train)
            self.best_params['catboost'] = params
        else:
            params = {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_seed': self.random_state,
                'verbose': False
            }
        
        model = CatBoostClassifier(**params)
        
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)
        
        self.models['catboost'] = model
        self.feature_importance['catboost'] = dict(zip(
            X_train.columns, model.feature_importances_
        ))
        
        return model
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.Series] = None,
                      method: str = 'voting') -> VotingClassifier:
        """Train ensemble model"""
        print(f"\nTraining {method} ensemble...")
        
        # Ensure base models are trained (only train models that exist in self.models)
        # Note: lightgbm has been removed, only train xgboost and catboost
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
        
        self.models['ensemble'] = ensemble
        
        return ensemble
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, model_name: str = None) -> Dict[str, float]:
        """Evaluate model(s)"""
        results = {}
        
        models_to_eval = {model_name: self.models[model_name]} if model_name else self.models
        
        for name, model in models_to_eval.items():
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            results[name] = {
                'roc_auc': roc_auc_score(y, y_pred_proba),
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred)
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"  ROC-AUC: {results[name]['roc_auc']:.4f}")
            print(f"  Accuracy: {results[name]['accuracy']:.4f}")
            print(f"  Precision: {results[name]['precision']:.4f}")
            print(f"  Recall: {results[name]['recall']:.4f}")
            print(f"  F1-Score: {results[name]['f1']:.4f}")
        
        return results
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """Cross-validation for all models"""
        print("\nPerforming cross-validation...")
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = {}
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            cv_scores[name] = scores
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        self.cv_scores = cv_scores
        return cv_scores
    
    def save_models(self, models_dir: Path):
        """Save all trained models"""
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = models_dir / f"{name}_model.pkl"
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save metadata
        metadata = {
            'best_params': self.best_params,
            'cv_scores': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                         for k, v in self.cv_scores.items()},
            'feature_importance': self.feature_importance
        }
        joblib.dump(metadata, models_dir / 'model_metadata.pkl')
    
    def load_model(self, model_path: Path, model_name: str):
        """Load a specific model"""
        model = joblib.load(model_path)
        self.models[model_name] = model
        return model
