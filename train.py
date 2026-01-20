"""Main training script with comprehensive reporting pipeline"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    
    # Load configuration
    config = load_config()
    paths = get_paths(config)
    
    # Setup logger
    logger = setup_logger('training', paths['logs'])
    logger.info("="*80)
    logger.info("Starting Insurance Renewal Prediction Training Pipeline")
    logger.info("="*80)
    
    # Load data
    logger.info(f"Loading data from {config['data']['train_path']}")
    df = pd.read_csv(config['data']['train_path'])
    logger.info(f"Data loaded: Shape {df.shape}")
    
    # Auto-discover features if not specified in config
    target_column = config['data']['target_column']
    discovered = discover_features(df, target_column=target_column)
    
    # Use discovered features or config features
    if config['features'].get('use_auto_discovery', False):
        numerical_features = discovered['numerical_features']
        categorical_features = discovered['categorical_features']
        logger.info(f"Using auto-discovered features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")
    else:
        numerical_features = config['features']['numerical_features']
        categorical_features = config['features']['categorical_features']
        logger.info(f"Using config features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")
    
    # EDA
    logger.info("Starting Exploratory Data Analysis...")
    eda_output_dir = paths['outputs'] / 'eda'
    explorer = DataExplorer(eda_output_dir)
    explorer.generate_report(df, target_column, numerical_features, categorical_features)
    logger.info(f"EDA completed. Reports saved to {eda_output_dir}")
    
    # Prepare data
    logger.info("Preparing data for training...")
    
    # Encode target
    le_target = LabelEncoder()
    df[target_column] = le_target.fit_transform(df[target_column])
    target_mapping = dict(zip(le_target.classes_, range(len(le_target.classes_))))
    logger.info(f"Target encoding: {target_mapping}")
    
    # Drop unnecessary columns
    drop_cols = config['features'].get('drop_features', [])
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    stratify = y if config['data'].get('stratify', True) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=stratify
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
    
    # Feature engineering
    logger.info("Starting feature engineering...")
    logger.info("IMPORTANT: Target encoding is DISABLED by default to prevent data leakage")
    logger.info("This ensures realistic AUC scores (0.75-0.85) instead of unrealistic 0.99+")
    
    # Target encoding disabled to prevent leakage in CV
    # If enabled, it would cause AUC > 0.99 because target encoding leaks in cross-validation
    use_target_encoding = config.get('features', {}).get('use_target_encoding', False)
    if use_target_encoding:
        logger.warning("Target encoding is ENABLED. This may cause data leakage in CV!")
        logger.warning("Ensure proper out-of-fold encoding is implemented.")
    else:
        logger.info("Target encoding is DISABLED (safe mode).")
    
    feature_engineer = FeatureEngineer(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_column=target_column,
        use_target_encoding=use_target_encoding
    )
    
    # Fit on training data (pass y_train separately to prevent data leakage)
    # This ensures target encoding only uses training data, not test data
    X_train_processed = feature_engineer.transform(X_train, y=y_train, fit=True)
    
    # Transform test data (no y needed - uses stored mappings from training)
    X_test_processed = feature_engineer.transform(X_test, fit=False)
    
    logger.info(f"Processed train set: {X_train_processed.shape}")
    logger.info(f"Processed test set: {X_test_processed.shape}")
    
    # Save feature engineer
    feature_engineer_path = paths['models'] / 'feature_engineer.pkl'
    feature_engineer.save(feature_engineer_path)
    logger.info(f"Feature engineer saved to {feature_engineer_path}")
    
    # Model training
    logger.info("Starting model training and optimization...")
    
    # Determine training mode (fast or full)
    training_mode = config['training'].get('mode', 'full')
    logger.info(f"Training mode: {training_mode.upper()}")
    
    if training_mode == 'fast':
        # Fast mode: fewer trials, fewer models, faster training
        fast_config = config['training'].get('fast', {})
        cv_folds = fast_config.get('cv_folds', 3)
        n_trials = fast_config.get('n_trials', 20)
        models_to_train = fast_config.get('models', ['lightgbm', 'ensemble'])
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
    
    trainer = ModelTrainer(
        cv_folds=cv_folds,
        random_state=config['data']['random_state'],
        scoring=config['model']['scoring'],
        n_trials=n_trials
    )
    
    logger.info(f"Training {len(models_to_train)} model(s): {models_to_train}")
    
    if 'lightgbm' in models_to_train:
        logger.info("Training LightGBM...")
        trainer.train_lightgbm(X_train_processed, y_train, optimize=True)
    
    if 'xgboost' in models_to_train:
        logger.info("Training XGBoost...")
        trainer.train_xgboost(X_train_processed, y_train, optimize=True)
    
    if 'catboost' in models_to_train:
        logger.info("Training CatBoost...")
        trainer.train_catboost(X_train_processed, y_train, optimize=True)
    
    if 'ensemble' in models_to_train:
        logger.info("Training Ensemble model...")
        trainer.train_ensemble(
            X_train_processed, y_train,
            method=config['model']['ensemble']['method']
        )
    
    # Cross-validation
    logger.info("Performing cross-validation...")
    cv_scores = trainer.cross_validate(X_train_processed, y_train)
    
    # Evaluate on test set
    logger.info("Evaluating models on test set...")
    test_results = trainer.evaluate(X_test_processed, y_test)
    
    # Get predictions and probabilities for reporting
    predictions = {}
    probabilities = {}
    for model_name, model in trainer.models.items():
        predictions[model_name] = model.predict(X_test_processed)
        probabilities[model_name] = model.predict_proba(X_test_processed)[:, 1]
    
    # Generate comprehensive report
    logger.info("Generating comprehensive training and evaluation report...")
    report_dir = paths['outputs'] / 'reports'
    report_generator = ModelReportGenerator(report_dir)
    
    training_time = time.time() - start_time
    
    # Generate training report
    training_report = report_generator.generate_training_report(
        trainer_results={},
        feature_importance=trainer.feature_importance,
        cv_scores=cv_scores,
        test_results=test_results,
        training_time=training_time
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
    
    logger.info("="*80)
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Total training time: {training_time/60:.2f} minutes ({training_time:.2f} seconds)")
    logger.info(f"Best model: {best_model}")
    logger.info(f"Best test ROC-AUC: {best_auc:.4f}")
    logger.info(f"Reports saved to: {report_dir}")
    logger.info(f"Models saved to: {paths['models']}")
    logger.info("="*80)
    
    return trainer, test_results, training_report


if __name__ == "__main__":
    trainer, results, report = main()
