# Insurance Renewal Prediction System

A State-of-the-Art (SOTA) machine learning system for predicting insurance customer renewal, built following Kaggle competition best practices.

## 🎯 Features

- **Comprehensive EDA**: Automated exploratory data analysis with visualizations
- **Advanced Feature Engineering**: Interaction features, statistical aggregations, target encoding following Kaggle best practices
- **Auto Feature Discovery**: Automatic detection of categorical and numerical features
- **Multiple SOTA Models**: LightGBM, XGBoost, CatBoost, and Ensemble
- **Hyperparameter Optimization**: Optuna-based automated tuning with cross-validation
- **Comprehensive Reporting**: Complete training, testing, and model accuracy reports
- **RESTful API**: FastAPI-based prediction service
- **Modular Design**: Clean, maintainable code structure

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
├── train.py              # Main training script
├── run_api.py            # API server script
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

## 📊 Model Performance & Reporting

The system generates comprehensive reports including:

### Training Report (`outputs/reports/training_report.json`)
- Cross-validation scores for all models
- Test set performance metrics
- Best model identification
- Feature importance rankings
- Training time statistics

### Visualizations (`outputs/reports/`)
- **model_performance.png**: ROC curves, PR curves, confusion matrix, metric comparison
- **feature_importance.png**: Top features for each model
- **cv_comparison.png**: Cross-validation score distributions

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
- Target encoding for categorical features
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
  train_path: "dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv"
  target_column: "Response"
  test_size: 0.2
  random_state: 42

features:
  use_auto_discovery: false  # Enable automatic feature discovery
  categorical_features: [...]
  numerical_features: [...]

model:
  models: ["lightgbm", "xgboost", "catboost", "ensemble"]
  cv_folds: 5
  n_trials: 100  # Hyperparameter optimization trials
```

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

## 🛠️ Dependencies

- **pandas, numpy**: Data manipulation
- **scikit-learn**: Machine learning utilities
- **lightgbm, xgboost, catboost**: Gradient boosting models
- **optuna**: Hyperparameter optimization
- **fastapi, uvicorn**: API framework
- **matplotlib, seaborn**: Visualization

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on the repository.
