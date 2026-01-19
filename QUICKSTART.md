# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Training

Run the complete training pipeline (includes EDA, feature engineering, model training, and reporting):

```bash
python train.py
```

This will:
- Perform EDA analysis and generate visualizations (saved in `outputs/eda/`)
- Engineer features
- Train and optimize multiple SOTA models (LightGBM, XGBoost, CatBoost, Ensemble)
- Generate comprehensive training and evaluation reports
- Evaluate model performance
- Save models and feature engineer (saved in `models/`)

**Note**: First run will perform hyperparameter optimization, which may take 30-60 minutes depending on hardware.

## Start API Service

After training, start the prediction API:

```bash
python run_api.py
```

API will be available at `http://localhost:8000`

## Test API

In another terminal, run:

```bash
python test_api.py
```

Or use curl:

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

## View API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## Configuration

Edit `config.yaml` to customize:

- **Model Selection**: Modify `model.models` list to choose which models to train
- **Optimization Trials**: Modify `model.n_trials` to adjust hyperparameter optimization rounds (reduce for faster training)
- **Cross-Validation Folds**: Modify `model.cv_folds`
- **API Port**: Modify `api.port`
- **Auto Feature Discovery**: Set `features.use_auto_discovery: true` to automatically discover features

## Performance Optimization Tips

1. **Quick Testing**: Set `n_trials` to 10-20 in `config.yaml`
2. **Full Training**: Use default 100 trials for best performance
3. **GPU Acceleration**: If you have GPU, set `use_gpu: true` in `config.yaml` (requires GPU-enabled libraries)

## Common Issues

### Q: Memory error during training
A: Reduce `n_trials` or `cv_folds`, or use fewer models

### Q: API cannot load models
A: Make sure to run `python train.py` first to train models

### Q: Prediction results inaccurate
A: Ensure input data format and ranges match training data

## Project Structure

```
InsurancePrediction/
├── src/              # Source code
├── dataset/          # Data files
├── models/           # Trained models (generated after training)
├── outputs/          # EDA outputs (generated after training)
│   ├── eda/          # EDA visualizations
│   └── reports/      # Training and evaluation reports
├── logs/             # Log files (generated after training)
├── config.yaml       # Configuration file
├── train.py          # Training script
└── run_api.py        # API service script
```
