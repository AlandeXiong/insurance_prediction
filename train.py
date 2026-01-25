"""Main training script with comprehensive reporting pipeline"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.utils.config import load_config, get_paths
from src.utils.logger import setup_logger
from src.utils.data_discovery import discover_features
from src.utils.reporting import ModelReportGenerator
from src.eda.explorer import DataExplorer
from src.features.engineering import FeatureEngineer
from src.models.trainer import ModelTrainer


def main():
    """
    Main training pipeline with comprehensive reporting.
    Follows Kaggle competition best practices.
    """
    start_time = time.time()

    # Step1 Load Training configuration
    config = load_config()
    paths = get_paths(config)

    # Setup logger
    logger = setup_logger('training', paths['logs'])
    logger.info("=" * 80)
    logger.info("Starting Insurance Renewal Prediction Training Pipeline")
    logger.info("=" * 80)

    # Step2 Load data - support both separate files and single file with split
    use_separate_files = config['data'].get('use_separate_files', False)
    target_column = config['data']['target_column']

    if use_separate_files:
        # Load separate train and test files
        train_path = config['data']['train_path']
        test_path = config['data'].get('test_path', None)

        if not test_path:
            raise ValueError("use_separate_files is True but test_path is not provided in config")

        logger.info(f"Loading training data from {train_path}")
        df_train = pd.read_csv(train_path)
        logger.info(f"Training data loaded: Shape {df_train.shape}")

        logger.info(f"Loading test data from {test_path}")
        df_test = pd.read_csv(test_path)
        logger.info(f"Test data loaded: Shape {df_test.shape}")

        # Use training data for EDA and feature discovery
        df = df_train.copy()

    else:
        # Load single file and split
        logger.info(f"Loading data from {config['data']['train_path']}")
        df = pd.read_csv(config['data']['train_path'])
        logger.info(f"Data loaded: Shape {df.shape}")

    # Step3 EDA Analysis Auto-discover features if not specified in config
    discovered = discover_features(df, target_column=target_column)

    # Use discovered features or config features
    if config['features'].get('use_auto_discovery', False):
        numerical_features = discovered['numerical_features']
        categorical_features = discovered['categorical_features']
        logger.info(
            f"Using auto-discovered features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")
    else:
        numerical_features = config['features']['numerical_features']
        categorical_features = config['features']['categorical_features']
        logger.info(
            f"Using config features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")

    # Generated EDA Report (using training data only)
    logger.info("Starting Exploratory Data Analysis...")
    eda_output_dir = paths['outputs'] / 'eda'
    explorer = DataExplorer(eda_output_dir)
    explorer.generate_report(df, target_column, numerical_features, categorical_features)
    logger.info(f"EDA completed. Reports saved to {eda_output_dir}")

    # Step4 Feature engineering and data preparation
    # Prepare data
    logger.info("Preparing data for training...")

    # 4.1 Encode target - fit on training data only
    le_target = LabelEncoder()

    # Determine threshold strategy
    model_cfg = config.get('model', {}) or {}
    threshold_strategy = str(model_cfg.get('threshold_strategy', 'fixed')).lower().strip()

    # Internal validation split (used for threshold selection when threshold_strategy="validation")
    df_train_fit = df_train.copy()
    df_val_fit = None
    if threshold_strategy == 'validation':
        val_cfg = (config.get('training', {}) or {}).get('validation', {}) or {}
        val_strategy = str(val_cfg.get('strategy', 'auto')).lower().strip()
        val_size = float(val_cfg.get('holdout_size', 0.2))
        date_col = str(val_cfg.get('date_col', 'Effective To Date'))

        def stratified_split_fallback():
            return train_test_split(
                df_train.copy(),
                test_size=val_size,
                random_state=config['data']['random_state'],
                stratify=df_train[target_column],
            )

        if (val_strategy in {'auto', 'time_holdout'} and date_col in df_train_fit.columns):
            dt = pd.to_datetime(df_train_fit[date_col], errors='coerce')
            df_train_fit = df_train_fit.assign(_dt=dt).sort_values('_dt').drop(columns=['_dt'])
            split_idx = int(round((1.0 - val_size) * len(df_train_fit)))
            split_idx = min(max(split_idx, 1), len(df_train_fit) - 1)
            df_val_fit = df_train_fit.iloc[split_idx:].copy()
            df_train_fit = df_train_fit.iloc[:split_idx].copy()
            # If time split produces single-class validation, fall back to stratified random split.
            if df_val_fit[target_column].nunique() < 2:
                logger.warning("Time-holdout validation produced single-class validation; falling back to stratified holdout.")
                df_train_fit, df_val_fit = stratified_split_fallback()
        else:
            df_train_fit, df_val_fit = stratified_split_fallback()

        logger.info(f"Validation split: train_fit={df_train_fit.shape}, val_fit={df_val_fit.shape}")

    # Process training data (fit encoder on train_fit only)
    df_train_fit[target_column] = le_target.fit_transform(df_train_fit[target_column])
    target_mapping = dict(zip(le_target.classes_, range(len(le_target.classes_))))
    logger.info(f"Target encoding (fitted on train): {target_mapping}")

    if df_val_fit is not None:
        df_val_fit[target_column] = le_target.transform(df_val_fit[target_column])

    # Process test data using same encoder
    # Handle unseen labels in test set
    test_targets = df_test[target_column].values
    test_targets_encoded = []
    unseen_labels = set()

    for val in test_targets:
        try:
            # Try to transform using the fitted encoder
            encoded = le_target.transform([val])[0]
            test_targets_encoded.append(encoded)
        except ValueError:
            # Unseen label in test set
            unseen_labels.add(val)
            # Use the first class from training set as default
            default_class = le_target.classes_[0]
            default_encoded = le_target.transform([default_class])[0]
            test_targets_encoded.append(default_encoded)
            logger.warning(
                f"Unseen target value '{val}' in test set. Using default encoding: {default_class} -> {default_encoded}")

    if unseen_labels:
        logger.warning(f"Found {len(unseen_labels)} unseen target value(s) in test set: {unseen_labels}")
        logger.warning("These have been encoded using the default class from training set.")

    df_test[target_column] = test_targets_encoded

    # 4.2 Process the X data
    # Drop unnecessary columns
    drop_cols = config['features'].get('drop_features', [])
    # If enabled, keep 'Effective To Date' for feature generation (month/day/weekday),
    # and drop the raw date column inside FeatureEngineer after deriving features.
    date_cfg = (config.get('features', {}) or {}).get('date_features', {}) or {}
    eff_cfg = (date_cfg.get('effective_to_date', {}) or {})
    eff_enabled = bool(eff_cfg.get('enabled', False))
    if eff_enabled and 'Effective To Date' in drop_cols:
        drop_cols = [c for c in drop_cols if c != 'Effective To Date']
    df_train_fit = df_train_fit.drop(columns=[col for col in drop_cols if col in df_train_fit.columns])
    if df_val_fit is not None:
        df_val_fit = df_val_fit.drop(columns=[col for col in drop_cols if col in df_val_fit.columns])
    df_test = df_test.drop(columns=[col for col in drop_cols if col in df_test.columns])

    # Split into X and y
    X_train = df_train_fit.drop(columns=[target_column])
    y_train = df_train_fit[target_column]
    X_val = None
    y_val = None
    if df_val_fit is not None:
        X_val = df_val_fit.drop(columns=[target_column])
        y_val = df_val_fit[target_column]
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]

    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
    if X_val is not None and y_val is not None:
        logger.info(f"Validation set: {X_val.shape}")
        logger.info(f"Validation target distribution: {y_val.value_counts().to_dict()}")


    # NOTE ABOUT FEATURE ENGINEERING (IMPORTANT)
    # This pipeline runs ONE unified feature engineering fit:
    # - Fit FeatureEngineer on FULL training set once
    # - Transform both train and test using the same fitted FeatureEngineer
    #
    # This is faster (especially in FAST mode) because we do not refit / re-transform
    # features inside every training loop. Thresholds are NOT auto-searched; they come from config.yaml.
    feature_engineer_kwargs = {
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "target_column": target_column,
    }

    # Fit feature engineering ONCE (fit on train_fit only)
    feature_engineer_final = FeatureEngineer(
        **feature_engineer_kwargs,
        drop_effective_to_date_original=bool(eff_cfg.get('drop_original', True)) if eff_enabled else True,
    )
    X_train_processed = feature_engineer_final.transform(X_train, y=y_train, fit=True)
    X_val_processed = None
    if X_val is not None:
        X_val_processed = feature_engineer_final.transform(X_val, fit=False)
    X_test_processed = feature_engineer_final.transform(X_test, fit=False)
    logger.info(f"Processed train set: {X_train_processed.shape}")
    if X_val_processed is not None:
        logger.info(f"Processed validation set: {X_val_processed.shape}")
    logger.info(f"Processed test set: {X_test_processed.shape}")

    # Save feature engineer (used by API/inference)
    feature_engineer_path = paths['models'] / 'feature_engineer.pkl'
    feature_engineer_final.save(feature_engineer_path)
    logger.info(f"Feature engineer saved to {feature_engineer_path}")

    logger.info("Starting model training and optimization...")

    # Determine training mode (fast or full)
    training_mode = config['training'].get('mode', 'full')
    logger.info(f"Training mode: {training_mode.upper()}")

    if training_mode == 'fast':
        # Fast mode: fewer trials, fewer models, faster training
        fast_config = config['training'].get('fast', {})
        cv_folds = fast_config.get('cv_folds', 3)
        n_trials = fast_config.get('n_trials', 20)
        models_to_train = fast_config.get('models', ['xgboost', 'ensemble'])
        # Fast tuning: single holdout split + smaller iteration budgets (2x+ speed)
        tuning_cfg = fast_config.get('tuning', {}) or {}
        tuning_strategy = str(tuning_cfg.get('strategy', 'holdout')).lower().strip()
        use_holdout_tuning = tuning_strategy == 'holdout'
        holdout_size = float(tuning_cfg.get('holdout_size', 0.2))
        tuning_early_stopping_rounds = int(tuning_cfg.get('early_stopping_rounds', 30))
        tuning_lgb_n_estimators = int(tuning_cfg.get('lightgbm_n_estimators', 800))
        tuning_xgb_n_estimators = int(tuning_cfg.get('xgboost_n_estimators', 600))
        tuning_cat_iterations = int(tuning_cfg.get('catboost_iterations', 600))
        tuning_cat_od_wait = int(tuning_cfg.get('catboost_od_wait', 30))
        logger.info("FAST MODE: Using reduced settings for quick training")
        logger.info(f"  CV folds: {cv_folds} (reduced from {config['model']['cv_folds']})")
        logger.info(f"  Optimization trials: {n_trials} (reduced from {config['model']['n_trials']})")
        logger.info(f"  Models: {models_to_train} (reduced set)")
        logger.info(
            f"  Tuning strategy: {'holdout' if use_holdout_tuning else 'kfold'} | "
            f"holdout_size={holdout_size} | "
            f"early_stopping_rounds={tuning_early_stopping_rounds}"
        )
        logger.info(
            f"  Tuning budgets: lgb_n_estimators={tuning_lgb_n_estimators}, "
            f"xgb_n_estimators={tuning_xgb_n_estimators}, "
            f"cat_iterations={tuning_cat_iterations}"
        )
    else:
        # Full mode: standard settings for best performance
        full_config = config['training'].get('full', {})
        cv_folds = full_config.get('cv_folds', config['model']['cv_folds'])
        n_trials = full_config.get('n_trials', config['model']['n_trials'])
        models_to_train = full_config.get('models', config['model']['models'])
        logger.info("FULL MODE: Using full settings for best performance")
        logger.info(f"  CV folds: {cv_folds}")
        logger.info(f"  Optimization trials: {n_trials}")
        logger.info(f"  Models: {models_to_train}")

    # Step5 Model Training

    min_recall = float(config.get('model', {}).get('min_recall', 0.5))
    # When not configured, default min_precision to 0.0 (only enforce recall)
    min_precision = float(config.get('model', {}).get('min_precision', 0.0) or 0.0)

    # Threshold selection: validation-based or fixed from config
    optimize_threshold_cfg = bool(config.get('model', {}).get('optimize_threshold', False))
    optimize_threshold = bool(optimize_threshold_cfg or (threshold_strategy == 'validation' and X_val_processed is not None))
    thresholds_cfg = config.get('model', {}).get('thresholds', {}) or {}
    default_threshold = float(thresholds_cfg.get('default', 0.5))
    model_thresholds = {
        k: float(v) for k, v in thresholds_cfg.items()
        if k not in {'default'} and v is not None
    }

    logger.info(f"Minimum recall requirement: {min_recall}")
    logger.info(f"Minimum precision requirement: {min_precision}")
    logger.info(f"Threshold strategy: {threshold_strategy}")
    logger.info(f"Threshold optimization (auto-search): {optimize_threshold} (config={optimize_threshold_cfg})")
    logger.info(f"Default threshold: {default_threshold}")
    logger.info(f"Per-model thresholds: {model_thresholds}")

    trainer = ModelTrainer(
        cv_folds=cv_folds,
        random_state=config['data']['random_state'],
        scoring=config['model']['scoring'],
        n_trials=n_trials,
        min_recall=min_recall,
        min_precision=min_precision,
        optimize_threshold=optimize_threshold,
        thresholds=model_thresholds,
        default_threshold=default_threshold,
    )

    logger.info(f"Training {len(models_to_train)} model(s): {models_to_train}")

    # All models use the same feature engineering pipeline (already applied)
    # All models will be trained with class weights and threshold optimization


    # 5.1 Hyperparameter optimization + final fit on FULL processed training data
    # Thresholds are NOT auto-searched; they are controlled via config.yaml.
    if 'lightgbm' in models_to_train:
        logger.info("Optimizing + training LightGBM on processed features...")
        trainer.best_params['lightgbm'] = trainer.optimize_lightgbm(
            X_train_processed,
            y_train,
            use_holdout=bool(use_holdout_tuning) if training_mode == 'fast' else False,
            holdout_size=float(holdout_size) if training_mode == 'fast' else 0.2,
            tuning_n_estimators=int(tuning_lgb_n_estimators) if training_mode == 'fast' else 800,
            tuning_early_stopping_rounds=int(tuning_early_stopping_rounds) if training_mode == 'fast' else 30,
        )
        trainer.train_lightgbm(
            X_train_processed, y_train, X_val_processed, y_val,
            optimize=False,
            params_override=trainer.best_params['lightgbm']
        )

    if 'xgboost' in models_to_train:
        logger.info("Optimizing + training XGBoost on processed features...")
        trainer.best_params['xgboost'] = trainer.optimize_xgboost(
            X_train_processed,
            y_train,
            use_holdout=bool(use_holdout_tuning) if training_mode == 'fast' else False,
            holdout_size=float(holdout_size) if training_mode == 'fast' else 0.2,
            tuning_n_estimators=int(tuning_xgb_n_estimators) if training_mode == 'fast' else 600,
            tuning_early_stopping_rounds=int(tuning_early_stopping_rounds) if training_mode == 'fast' else 30,
        )
        trainer.train_xgboost(
            X_train_processed, y_train, X_val_processed, y_val,
            optimize=False,
            params_override=trainer.best_params['xgboost']
        )

    if 'catboost' in models_to_train:
        logger.info("Optimizing + training CatBoost on processed features...")
        trainer.best_params['catboost'] = trainer.optimize_catboost(
            X_train_processed,
            y_train,
            cat_feature_names=categorical_features,
            use_holdout=bool(use_holdout_tuning) if training_mode == 'fast' else False,
            holdout_size=float(holdout_size) if training_mode == 'fast' else 0.2,
            tuning_iterations=int(tuning_cat_iterations) if training_mode == 'fast' else 600,
            tuning_od_wait=int(tuning_cat_od_wait) if training_mode == 'fast' else 30,
        )
        trainer.train_catboost(
            X_train_processed, y_train, X_val_processed, y_val,
            optimize=False,
            cat_feature_names=categorical_features,
            params_override=trainer.best_params['catboost']
        )

    if 'ensemble' in models_to_train:
        logger.info("Training Ensemble model...")
        trainer.train_ensemble(
            X_train_processed, y_train, X_val_processed, y_val,
            method=config['model']['ensemble']['method'],
            base_models=[m for m in models_to_train if m != "ensemble"]
        )

    # Optional CV reporting (skip in FAST mode for speed)
    if training_mode == 'fast':
        cv_scores = {}
    else:
        cv_scores = trainer.cross_validate(X_train_processed, y_train)

    # Step6 Select BEST MODEL (avoid selecting on test set)
    model_cfg = config.get('model', {}) or {}
    best_model_metric = str(model_cfg.get('best_model_metric', 'business_score')).lower().strip()
    best_model_selection_dataset = str(model_cfg.get('best_model_selection_dataset', 'validation')).lower().strip()
    best_model_fallback_metric = str(model_cfg.get('best_model_fallback_metric', 'pr_auc')).lower().strip()

    def _eval_binary_metrics(y_true_arr: np.ndarray, proba_arr: np.ndarray, threshold_val: float) -> dict:
        y_pred_arr = (proba_arr >= float(threshold_val)).astype(int)
        prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr))
        business = prec if (rec >= float(min_recall) and prec >= float(min_precision)) else 0.0
        return {
            "roc_auc": float(roc_auc_score(y_true_arr, proba_arr)),
            "pr_auc": float(average_precision_score(y_true_arr, proba_arr)),
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
            "precision": prec,
            "recall": rec,
            "f1": float(f1_score(y_true_arr, y_pred_arr)),
            "business_score": float(business),
            "threshold": float(threshold_val),
        }

    selection_results = {}
    selection_best_model = None
    selection_best_value = None
    selection_source = None

    if best_model_selection_dataset == "validation" and X_val_processed is not None and y_val is not None:
        logger.info("Selecting BEST MODEL on validation set (to avoid test-set model selection)...")
        selection_source = "validation"
        for model_name in list(trainer.models.keys()):
            proba_val = trainer.models[model_name].predict_proba(X_val_processed)[:, 1]
            thr = float(trainer.best_thresholds.get(model_name, trainer.default_threshold))
            selection_results[model_name] = _eval_binary_metrics(y_val.values, proba_val, thr)

        # Primary: use configured metric; if business_score and none satisfy constraints, fall back.
        if best_model_metric == "business_score":
            feasible = {k: v for k, v in selection_results.items() if float(v.get("business_score", 0.0)) > 0.0}
            pool = feasible if feasible else selection_results
            metric_for_rank = "business_score" if feasible else best_model_fallback_metric
        else:
            pool = selection_results
            metric_for_rank = best_model_metric

        selection_best_model = max(
            pool.keys(),
            key=lambda m: float(pool[m].get(metric_for_rank, pool[m].get("roc_auc", 0.0))),
        ) if pool else None
        selection_best_value = float(pool[selection_best_model].get(metric_for_rank, 0.0)) if selection_best_model else None

        if best_model_metric == "business_score" and not any(float(v.get("business_score", 0.0)) > 0.0 for v in selection_results.values()):
            logger.warning(
                "No model satisfied constraints on validation set "
                f"(precision>={min_precision}, recall>={min_recall}). "
                f"Falling back to metric='{metric_for_rank}' for BEST MODEL selection."
            )
    else:
        selection_source = "test"
        logger.warning(
            "BEST MODEL selection will fall back to test-set metrics "
            f"(selection_dataset='{best_model_selection_dataset}', val_available={X_val_processed is not None}). "
            "For leakage-free evaluation, set model.best_model_selection_dataset='validation' and enable validation split."
        )

    # Step7 Evaluate models on the test set (final holdout evaluation)
    logger.info("Evaluating models on test set (using config thresholds)...")
    test_results = {}
    predictions = {}
    probabilities = {}
    models_for_eval = list(trainer.models.keys())
    for model_name in models_for_eval:
        proba = trainer.models[model_name].predict_proba(X_test_processed)[:, 1]
        threshold = float(trainer.best_thresholds.get(model_name, trainer.default_threshold))
        y_pred = (proba >= threshold).astype(int)
        y_pred_default = (proba >= 0.5).astype(int)
        # Business score (strict): precision at the operating point ONLY if constraints are satisfied.
        # Constraints: recall >= min_recall AND precision >= min_precision
        precision_at_threshold = float(precision_score(y_test, y_pred, zero_division=0))
        recall_at_threshold = float(recall_score(y_test, y_pred))
        business_score = float(precision_at_threshold) if (
            recall_at_threshold >= float(min_recall) and precision_at_threshold >= float(min_precision)
        ) else 0.0

        test_results[model_name] = {
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "accuracy": accuracy_score(y_test, y_pred),
            # Balanced accuracy is more informative than accuracy on imbalanced data.
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "business_score": business_score,
            "threshold": float(threshold),
            "accuracy_default": accuracy_score(y_test, y_pred_default),
            "balanced_accuracy_default": balanced_accuracy_score(y_test, y_pred_default),
            "precision_default": precision_score(y_test, y_pred_default, zero_division=0),
            "recall_default": recall_score(y_test, y_pred_default),
            "f1_default": f1_score(y_test, y_pred_default),
        }

        predictions[model_name] = y_pred
        probabilities[model_name] = proba

    # Step6 Collect the report
    # Generate comprehensive report
    logger.info("Generating comprehensive training and evaluation report...")
    report_dir = paths['outputs'] / 'reports'
    report_generator = ModelReportGenerator(report_dir)

    training_time = time.time() - start_time

    # Generate a training report
    training_report = report_generator.generate_training_report(
        trainer_results={},
        feature_importance=trainer.feature_importance,
        cv_scores=cv_scores,
        test_results=test_results,
        training_time=training_time,
        cv_metric_name=str(config['model']['scoring']),
        best_model_metric=best_model_metric,
        best_model_override=selection_best_model,
        best_model_selection={
            "dataset": selection_source,
            "metric": best_model_metric,
            "fallback_metric": best_model_fallback_metric,
            "value": selection_best_value,
        },
    )

    # Generate visualizations
    logger.info("Generating performance visualizations...")
    report_generator.generate_performance_plots(
        y_test.values,
        predictions,
        probabilities
    )

    report_generator.generate_feature_importance_plots(
        trainer.feature_importance,
        top_n=20
    )

    if cv_scores:
        report_generator.generate_cv_comparison_plot(cv_scores)

    # Generate text summary
    summary_report = report_generator.generate_summary_report()
    logger.info("\n" + summary_report)

    # Save JSON report
    report_generator.save_report_json('training_report.json')

    # Save models
    logger.info("Saving trained models...")
    trainer.save_models(paths['models'])

    # Save target encoder
    import joblib
    joblib.dump(le_target, paths['models'] / 'target_encoder.pkl')

    # Final summary
    best_model = training_report['best_model']
    best_auc = float(test_results[best_model]['roc_auc']) if best_model else 0.0
    best_pr_auc = float(test_results[best_model].get('pr_auc', 0.0)) if best_model else 0.0
    best_business = float(test_results[best_model].get('business_score', 0.0)) if best_model else 0.0

    logger.info("=" * 80)
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Total training time: {training_time / 60:.2f} minutes ({training_time:.2f} seconds)")
    logger.info(f"Best model: {best_model} (selected_on={selection_source}, metric={best_model_metric})")
    logger.info(f"Best test ROC-AUC: {best_auc:.4f} | PR-AUC: {best_pr_auc:.4f} | Business_score: {best_business:.4f}")
    logger.info(f"Reports saved to: {report_dir}")
    logger.info(f"Models saved to: {paths['models']}")
    logger.info("=" * 80)

    return trainer, test_results, training_report


if __name__ == "__main__":
    trainer, results, report = main()
