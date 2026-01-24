"""Main training script with comprehensive reporting pipeline"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

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

    # Process training data
    df_train[target_column] = le_target.fit_transform(df_train[target_column])
    target_mapping = dict(zip(le_target.classes_, range(len(le_target.classes_))))
    logger.info(f"Target encoding (fitted on train): {target_mapping}")

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
    df_train = df_train.drop(columns=[col for col in drop_cols if col in df_train.columns])
    df_test = df_test.drop(columns=[col for col in drop_cols if col in df_test.columns])

    # Split into X and y
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]

    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")


    # NOTE ABOUT FEATURE ENGINEERING (IMPORTANT)
    # This training pipeline intentionally uses feature engineering in two different "fit scopes":
    #
    # 1) TUNING scope (train-fit/val-fit):
    #    - We split RAW data into train-fit and val-fit first.
    #    - We fit FeatureEngineer on train-fit ONLY, then transform both train-fit and val-fit.
    #    - This prevents leakage of statistics (scaler/encoders/group aggregates) into val-fit.
    #    - Hyperparameter tuning and early stopping are done using these processed matrices.
    #
    # 2) FINAL/INFERENCE scope (full train / test):
    #    - We fit a new FeatureEngineer on the FULL training set, then transform train and test.
    #    - We save this FeatureEngineer to disk for API/inference to guarantee consistent preprocessing.
    #
    # We keep these two scopes separate on purpose: it is best practice for robust evaluation.
    feature_engineer_kwargs = {
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "target_column": target_column,
    }

    # Split BEFORE feature engineering (prevents leakage into validation metrics/thresholds)
    from sklearn.model_selection import train_test_split as tts
    X_train_fit_raw, X_val_fit_raw, y_train_fit, y_val_fit = tts(
        X_train, y_train,
        test_size=0.2,
        random_state=config['data']['random_state'],
        stratify=y_train
    )
    logger.info(f"Training set for model fitting (raw): {X_train_fit_raw.shape}")
    logger.info(f"Validation set for threshold optimization (raw): {X_val_fit_raw.shape}")

    # 4.3 Fit feature engineering
    # Fit feature engineering on TRAIN-FIT only, transform TRAIN-FIT and VAL-FIT
    feature_engineer_tune = FeatureEngineer(**feature_engineer_kwargs)
    X_train_fit = feature_engineer_tune.transform(X_train_fit_raw, y=y_train_fit, fit=True)
    X_val_fit = feature_engineer_tune.transform(X_val_fit_raw, fit=False)
    logger.info(f"Training set for model fitting (processed): {X_train_fit.shape}")
    logger.info(f"Validation set for threshold optimization (processed): {X_val_fit.shape}")

    # From this point, ALL models in the tuning stage use the SAME processed matrices
    # (X_train_fit / X_val_fit). This is the unified feature pipeline across models.
    # Model training (after feature engineering is prepared)
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
        logger.info("FAST MODE: Using reduced settings for quick training")
        logger.info(f"  CV folds: {cv_folds} (reduced from {config['model']['cv_folds']})")
        logger.info(f"  Optimization trials: {n_trials} (reduced from {config['model']['n_trials']})")
        logger.info(f"  Models: {models_to_train} (reduced set)")
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

    min_recall = config.get('model', {}).get('min_recall', 0.5)
    min_precision = config.get('model', {}).get('min_precision', 0.3)
    optimize_threshold = config.get('model', {}).get('optimize_threshold', True)
    threshold_selection = config.get('model', {}).get('threshold_selection', 'max_precision')

    logger.info(f"Minimum recall requirement: {min_recall}")
    logger.info(f"Minimum precision requirement: {min_precision}")
    logger.info(f"Threshold optimization: {optimize_threshold}")
    logger.info(f"Threshold selection strategy: {threshold_selection}")

    trainer = ModelTrainer(
        cv_folds=cv_folds,
        random_state=config['data']['random_state'],
        scoring=config['model']['scoring'],
        n_trials=n_trials,
        min_recall=min_recall,
        min_precision=min_precision,
        optimize_threshold=optimize_threshold,
        threshold_selection=threshold_selection
    )

    logger.info(f"Training {len(models_to_train)} model(s): {models_to_train}")

    # All models use the same feature engineering pipeline (already applied)
    # All models will be trained with class weights and threshold optimization


    # 5.1 First Round Training

    if 'xgboost' in models_to_train:
        logger.info("Training XGBoost...")
        trainer.train_xgboost(X_train_fit, y_train_fit, X_val_fit, y_val_fit, optimize=True)

    if 'lightgbm' in models_to_train:
        logger.info("Training LightGBM...")
        trainer.train_lightgbm(X_train_fit, y_train_fit, X_val_fit, y_val_fit, optimize=True)

    if 'catboost' in models_to_train:
        logger.info("Training CatBoost...")
        trainer.train_catboost(
            X_train_fit, y_train_fit, X_val_fit, y_val_fit,
            optimize=True,
            cat_feature_names=categorical_features
        )

    if 'ensemble' in models_to_train:
        logger.info("Training Ensemble model...")
        trainer.train_ensemble(
            X_train_fit, y_train_fit, X_val_fit, y_val_fit,
            method=config['model']['ensemble']['method'],
            base_models=[m for m in models_to_train if m != "ensemble"]
        )

    # 5.2 Fit FINAL feature engineering on FULL TRAIN and transform TRAIN/TEST for final training/inference
    # NOTE: This is the SECOND scope described above (final/inference scope).
    feature_engineer_final = FeatureEngineer(**feature_engineer_kwargs)
    X_train_processed = feature_engineer_final.transform(X_train, y=y_train, fit=True)
    X_test_processed = feature_engineer_final.transform(X_test, fit=False)
    logger.info(f"Processed train set (final): {X_train_processed.shape}")
    logger.info(f"Processed test set (final): {X_test_processed.shape}")

    # Save FINAL feature engineer (used by API/inference)
    feature_engineer_path = paths['models'] / 'feature_engineer.pkl'
    feature_engineer_final.save(feature_engineer_path)
    logger.info(f"Feature engineer saved to {feature_engineer_path}")

    # 5.3 First Refit final models
    # Refit final models on FULL training data using best params (Kaggle best practice)
    logger.info("Refitting final models on full training data with best hyperparameters...")
    if 'lightgbm' in trainer.best_params:
        trainer.train_lightgbm(
            X_train_processed, y_train, None, None,
            optimize=False,
            params_override=trainer.best_params['lightgbm']
        )
    if 'xgboost' in trainer.best_params:
        trainer.train_xgboost(
            X_train_processed, y_train, None, None,
            optimize=False,
            params_override=trainer.best_params['xgboost']
        )
    if 'catboost' in models_to_train and 'catboost' in trainer.best_params:
        trainer.train_catboost(
            X_train_processed, y_train, None, None,
            optimize=False,
            cat_feature_names=categorical_features,
            params_override=trainer.best_params['catboost']
        )
    if 'ensemble' in models_to_train:
        trainer.train_ensemble(
            X_train_processed, y_train, None, None,
            method=config['model']['ensemble']['method'],
            base_models=[m for m in models_to_train if m != "ensemble"]
        )

    # 5.4 Fit robust OOF thresholds on RAW training data (feature engineering fit per fold)
    # IMPORTANT: This also stores fold models + fold feature engineers for bagging inference.
    models_for_eval = list(trainer.models.keys())
    logger.info(f"Fitting robust OOF thresholds on training data (leakage-free) for models: {models_for_eval}")
    oof_threshold_metrics = trainer.fit_oof_thresholds_raw(
        X_train,
        y_train,
        feature_engineer_kwargs=feature_engineer_kwargs,
        model_names=models_for_eval,
        cat_feature_names=categorical_features
    )
    for m, met in oof_threshold_metrics.items():
        logger.info(
            f"OOF threshold for {m}: {met.get('threshold'):.4f} | "
            f"precision={met.get('precision'):.4f}, recall={met.get('recall'):.4f}, f1={met.get('f1'):.4f}"
        )

    # 5.5 Leakage-free cross-validation (feature engineering fit per fold)
    logger.info("Performing leakage-free cross-validation (feature engineering fit per fold)...")
    cv_scores = trainer.cross_validate_raw(
        X_train,
        y_train,
        feature_engineer_kwargs=feature_engineer_kwargs,
        model_names=models_for_eval,
        cat_feature_names=categorical_features
    )

    # Step6 Evaluate models on the test set
    # Evaluate on a test set (NO test-label leakage)
    # IMPORTANT:
    # - Thresholds are selected on TRAIN via OOF predictions.
    # - Test probabilities must be produced by the SAME fold-bagging process to avoid
    #   distribution mismatch (otherwise recall may collapse to 0).
    logger.info("Evaluating models on test set (using OOF-selected thresholds + fold-bagging probabilities)...")
    test_results = {}
    predictions = {}
    probabilities = {}
    for model_name in models_for_eval:
        # Bagged probabilities from fold models
        proba = trainer.predict_proba_bagged_raw(X_test, model_name)
        threshold = trainer.best_thresholds.get(model_name, 0.5)
        y_pred = (proba >= threshold).astype(int)
        y_pred_default = (proba >= 0.5).astype(int)

        test_results[model_name] = {
            "roc_auc": roc_auc_score(y_test, proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "threshold": float(threshold),
            "accuracy_default": accuracy_score(y_test, y_pred_default),
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
        cv_metric_name=f"Precision@Recall>={min_recall}",
        best_model_metric="roc_auc"
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
    best_auc = test_results[best_model]['roc_auc'] if best_model else 0

    logger.info("=" * 80)
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Total training time: {training_time / 60:.2f} minutes ({training_time:.2f} seconds)")
    logger.info(f"Best model: {best_model}")
    logger.info(f"Best test ROC-AUC: {best_auc:.4f}")
    logger.info(f"Reports saved to: {report_dir}")
    logger.info(f"Models saved to: {paths['models']}")
    logger.info("=" * 80)

    return trainer, test_results, training_report


if __name__ == "__main__":
    trainer, results, report = main()
