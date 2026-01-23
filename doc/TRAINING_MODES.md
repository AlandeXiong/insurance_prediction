# Training Modes Documentation

## Overview

The system now supports two training modes for flexibility:

1. **Fast Mode**: Quick testing and development
2. **Full Mode**: Production-ready training with best performance

## Configuration

Edit `config.yaml` to set the training mode:

```yaml
training:
  mode: "full"  # "fast" or "full"
  
  # Fast mode settings
  fast:
    n_trials: 20
    cv_folds: 3
    models: ["lightgbm", "ensemble"]
  
  # Full mode settings
  full:
    n_trials: 100
    cv_folds: 5
    models: ["lightgbm", "xgboost", "catboost", "ensemble"]
```

## Fast Mode

**Purpose**: Quick testing, development, and experimentation

**Settings**:
- **Optimization trials**: 20 (vs 100 in full mode)
- **CV folds**: 3 (vs 5 in full mode)
- **Models**: Only LightGBM + Ensemble (vs all 4 models)

**Time**: ~5-10 minutes (vs 30-60 minutes in full mode)

**Use cases**:
- Initial development and testing
- Quick feature engineering experiments
- Code debugging
- Learning and experimentation

**Performance**: Good but not optimal (~2-5% lower AUC than full mode)

## Full Mode

**Purpose**: Production training with best performance

**Settings**:
- **Optimization trials**: 100 (comprehensive hyperparameter search)
- **CV folds**: 5 (standard cross-validation)
- **Models**: All models (LightGBM, XGBoost, CatBoost, Ensemble)

**Time**: ~30-60 minutes (depending on hardware)

**Use cases**:
- Final model training
- Production deployment
- Competition submissions
- Research publications

**Performance**: Optimal performance with best hyperparameters

## Switching Modes

Simply change `training.mode` in `config.yaml`:

```yaml
# For quick testing
training:
  mode: "fast"

# For production
training:
  mode: "full"
```

## Customization

You can customize each mode's settings:

```yaml
training:
  mode: "fast"
  
  fast:
    n_trials: 10      # Even faster
    cv_folds: 3
    models: ["lightgbm"]  # Only one model
    
  full:
    n_trials: 200     # More thorough search
    cv_folds: 10      # More CV folds
    models: ["lightgbm", "xgboost", "catboost", "ensemble"]
```

## Recommendations

1. **Development Phase**: Use `fast` mode for quick iterations
2. **Final Training**: Use `full` mode before deployment
3. **Experimentation**: Start with `fast`, switch to `full` when satisfied
4. **Production**: Always use `full` mode

## Performance Comparison

Typical performance differences:

| Metric | Fast Mode | Full Mode | Difference |
|--------|-----------|-----------|------------|
| AUC | ~0.82 | ~0.85 | +3% |
| Training Time | 5-10 min | 30-60 min | 5-6x longer |
| Models Trained | 2 | 4 | 2x more |
| Optimization Trials | 20 | 100 | 5x more |

*Note: Actual performance depends on dataset and hardware*
