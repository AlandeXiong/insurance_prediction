# Insurance Renewal Prediction System

A State-of-the-Art (SOTA) machine learning system for predicting insurance customer renewal/repurchase, built following Kaggle-style competition best practices (clean splits, leakage prevention, CV + Optuna, strong boosting models, and threshold moving for imbalanced data).

## 🎯 Features

- **Comprehensive EDA**: Automated exploratory data analysis with visualizations
- **Advanced Feature Engineering**: Interaction features + safe statistical aggregations (count/median encodings)
- **Auto Feature Discovery**: Automatic detection of categorical and numerical features
- **Multiple SOTA Models**: LightGBM, XGBoost, CatBoost, and Ensemble
- **Hyperparameter Optimization**: Optuna-based automated tuning with cross-validation
- **Comprehensive Reporting**: Complete training, testing, and model performance reports (ROC-AUC + PR-AUC + threshold metrics)
- **RESTful API**: FastAPI-based prediction service
- **Modular Design**: Clean, maintainable code structure
- **Imbalanced Learning Best Practice**: Class weights + threshold moving with explicit constraints (**recall ≥ min_recall** and **precision ≥ min_precision**)
- **Leakage-safe best model selection**: Best model is selected on **validation** by default (test set is a final holdout)

## 📁 Project Structure

```
InsurancePrediction/
├── src/
│   ├── eda/              # Exploratory Data Analysis
│   │   └── explorer.py
│   ├── features/         # Feature Engineering
│   │   └── engineering.py
│   ├── models/           # Model Training
│   │   └── trainer.py
│   ├── api/              # API Service
│   │   ├── app.py
│   │   └── predictor.py
│   └── utils/            # Utilities
│       ├── config.py
│       ├── data_loading.py
│       ├── logger.py
│       ├── data_discovery.py
│       └── reporting.py
├── dataset/              # Data files
├── models/               # Saved models
├── outputs/              # EDA outputs and reports
│   ├── eda/              # EDA visualizations
│   └── reports/          # Training and evaluation reports
├── logs/                 # Training logs
├── config.yaml           # Configuration file
├── Dockerfile            # Container image for API service
├── docker-compose.yml    # One-command deployment (API)
├── train.py              # Main training script
├── run_api.py            # API server script
├── test_api.py           # Simple API smoke tests
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd InsurancePrediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.yaml` to customize:
- Data paths
- Feature lists (or enable auto-discovery)
- Model parameters
- API settings

**Key Configuration Options:**
- `features.use_auto_discovery`: Set to `true` to automatically discover features
- `model.n_trials`: Number of hyperparameter optimization trials (default: 100)
- `model.cv_folds`: Number of cross-validation folds (default: 5)

### 3. Training

```bash
# Run the complete training pipeline
python train.py
```

This will:
- Perform EDA and generate visualizations
- Engineer features (with auto-discovery if enabled)
- Train and optimize multiple models
- Generate comprehensive reports
- Evaluate performance
- Save models and artifacts

**Outputs:**
- EDA reports: `outputs/eda/`
- Training reports: `outputs/reports/`
- Models: `models/`
- Logs: `logs/`

### 4. API Service

```bash
# Start the prediction API
python run_api.py
```

The API will be available at `http://localhost:8000`

### 5. Docker Compose Deployment (Recommended for serving)

**Prerequisite**: you must have trained artifacts in `./models/` (at least `feature_engineer.pkl` and one `*_model.pkl`).

```bash
# 1) Train (generates ./models/*)
python train.py

# 2) Start API service
docker compose up --build
```

The compose service:
- Mounts `./models` into the container at `/app/models`
- Mounts `./config.yaml` into the container at `/app/config.yaml` (read-only)
- Exposes API on `http://localhost:8000`

### 6. API Testing (local or Docker)

**Option A: curl smoke tests**

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/models"
```

**Option B: run the provided test script**

```bash
python test_api.py
```

## 📊 Model Performance & Reporting

The system generates comprehensive reports including:

### Training Report (`outputs/reports/training_report.json`)
- Cross-validation scores for all models
- Test set performance metrics (includes **PR-AUC**)
- Best model identification
- Feature importance rankings
- Training time statistics

### Visualizations (`outputs/reports/`)
- **model_performance.png**: ROC curves, PR curves, confusion matrix, metric comparison
- **feature_importance.png**: Top features for each model
- **cv_comparison.png**: Cross-validation score distributions
- **model_structure/** (optional, `analysis.model_structure.enabled: true`): Tree structure plots for LightGBM/XGBoost/CatBoost (first N trees per model) and per-model feature importance bar charts

### Text Summary (`outputs/reports/training_report.txt`)
- Human-readable summary of all results
- Best model performance
- Key metrics comparison

## 🔌 API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "State": "California",
    "Coverage": "Premium",
    "Education": "Bachelor",
    "EmploymentStatus": "Employed",
    "Gender": "M",
    "Income": 50000,
    "Location Code": "Suburban",
    "Marital Status": "Married",
    "Monthly Premium Auto": 100,
    "Months Since Last Claim": 12,
    "Months Since Policy Inception": 24,
    "Number of Open Complaints": 0,
    "Number of Policies": 1,
    "Policy Type": "Personal Auto",
    "Policy": "Personal L3",
    "Renew Offer Type": "Offer1",
    "Sales Channel": "Agent",
    "Total Claim Amount": 500,
    "Vehicle Class": "Four-Door Car",
    "Vehicle Size": "Medsize"
  }'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[{...}, {...}]'
```

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## 🎨 Feature Engineering

The system includes advanced feature engineering following Kaggle best practices:

### Interaction Features
- Premium per policy
- Claim frequency
- Average claim amount
- Customer lifetime value to income ratio
- Premium to income ratio

### Statistical Features
- Count encoding
- Median encoding (numerical features grouped by categories)

### Time-based Features
- Recent claim indicators
- Policy age in years
- Long-term customer flags
- Claim recency scores

### Transformations
- Logarithmic transformations for skewed features
- Standard scaling for numerical features
- Label encoding for categorical features

## 📈 EDA Outputs

The EDA module generates:
- Target distribution analysis
- Numerical feature distributions
- Categorical feature analysis
- Correlation matrices
- Feature-target relationships

All visualizations are saved in `outputs/eda/`

## 🔧 Configuration

Key configuration options in `config.yaml`:

```yaml
data:
  # Data split strategy (ONLY 3 supported):
  # - pre_split: use already-split train/test files on disk
  # - timecut: split the original source file by a date cutoff (time-based holdout)
  # - ratio: random split the original source file by test_size (optionally stratified)
  strategy: "pre_split"  # "pre_split" | "timecut" | "ratio"

  target_column: "Response"
  random_state: 42
  source_path: "dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv"

  # Strategy 1) pre_split
  pre_split:
    train_path: "dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis—train_timecut.csv"
    test_path: "dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis—test_timecut.csv"

  # Strategy 2) timecut (split from source_path)
  timecut:
    date_col: "Effective To Date"
    cutoff: "2011-02-20"

  # Strategy 3) ratio (random split from source_path)
  ratio:
    test_size: 0.2
    stratify: true

  # Optional: deduplicate by feature columns before split; save unique + duplicates CSVs under outputs
  deduplicate:
    enabled: false
    export_dir: "outputs/deduplicated"
    export_unique_name: "unique.csv"
    export_duplicates_name: "duplicates.csv"

model:
  models: ["lightgbm", "xgboost", "catboost", "ensemble"]
  cv_folds: 5
  n_trials: 100
  scoring: "roc_auc"   # Optuna/CV scoring: "roc_auc" or "average_precision"
  min_recall: 0.5
  # min_precision: 0.25

  # Threshold selection:
  threshold_strategy: "validation"  # "fixed" | "validation"
  thresholds:
    default: 0.5
    lightgbm: 0.30
    xgboost: 0.34
    catboost: 0.38
    ensemble: 0.35

  # Best model selection (leakage-safe by default):
  best_model_selection_dataset: "validation"   # "validation" | "test"
  best_model_metric: "business_score"          # "business_score" | "pr_auc" | "roc_auc" | ...
  best_model_fallback_metric: "pr_auc"

training:
  mode: "fast"  # "fast" or "full"
  fast:
    n_trials: 20
    cv_folds: 3
    models: ["lightgbm", "xgboost", "ensemble"]
  full:
    n_trials: 100
    cv_folds: 5
    models: ["lightgbm", "xgboost", "catboost", "ensemble"]
```

### Training Modes

- **Fast Mode**: Quick testing and development (~5-10 min)
  - Fewer optimization trials (20)
  - Fewer CV folds (3)
  - Fewer models (XGBoost + Ensemble)
  
- **Full Mode**: Production training (~30-60 min)
  - Full optimization (100 trials)
  - Standard CV (5 folds)
  - All models (XGBoost, CatBoost, Ensemble)

See `doc/TRAINING_MODES.md` for details.

## 📝 Best Practices

This project follows Kaggle competition best practices:

1. **Modular Code Structure**: Separated concerns (EDA, features, models, API)
2. **Configuration Management**: YAML-based config for easy experimentation
3. **Comprehensive Logging**: Detailed logs for debugging and tracking
4. **Cross-Validation**: Stratified K-Fold for robust evaluation
5. **Hyperparameter Optimization**: Automated tuning with Optuna
6. **Model Ensemble**: Combining multiple models for better performance
7. **Feature Engineering**: Advanced techniques following Kaggle patterns
8. **Comprehensive Reporting**: Complete training and evaluation reports
9. **Auto Feature Discovery**: Automatic feature identification
10. **API Design**: RESTful API with proper error handling
11. **Data Leakage Prevention**: Target encoding uses only training data
12. **Flexible Training Modes**: Fast mode for development, full mode for production

## 🔒 Data Leakage Prevention

The system implements strict data leakage prevention:

- **No test-label tuning**: thresholds are selected on validation data and applied to the test set
- **No test-set model selection (default)**: best model is selected on validation metrics by default
- **Automatic Checks**: Target column automatically removed if accidentally included in features
- **Explicit Separation**: Clear separation between feature data and target data
- **Training/Test Isolation**: Test data never used for feature engineering

See `doc/DATA_LEAKAGE_FIX.md` for details.

## 🛠️ Dependencies

- **pandas, numpy**: Data manipulation
- **scikit-learn**: Machine learning utilities
- **lightgbm, xgboost, catboost**: Gradient boosting models

### macOS (Apple Silicon) note

On Mac M1/M2, LightGBM normally installs via prebuilt wheels. If you hit an OpenMP error at runtime, install OpenMP:

```bash
brew install libomp
```
- **optuna**: Hyperparameter optimization
- **fastapi, uvicorn**: API framework
- **matplotlib, seaborn**: Visualization

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on the repository.
