# Data Leakage Analysis and Fix

## Problem Identified

The code has a **critical data leakage issue** in target encoding that can cause AUC to be artificially high (close to 1).

## Root Cause

### Issue 1: Target Encoding Without Proper Cross-Validation

In `src/features/engineering.py`, the `create_statistical_features` method performs target encoding:

```python
if fit and self.target_column in df.columns:
    target_mean = df.groupby(cat_col)[self.target_column].mean()
    self.target_encoding_maps[cat_col] = target_mean.to_dict()
    df[f'{cat_col}_Target_Mean'] = df[cat_col].map(target_mean)
```

**Problem**: When this is used in cross-validation or on the training set, it calculates the target mean using **all** data in the current fold/split, including the sample being predicted. This causes data leakage.

### Issue 2: Training Pipeline Issue

In `train.py` line 99:
```python
X_train_processed = feature_engineer.transform(X_train, fit=True)
```

**Problem**: `X_train` doesn't contain the target column, so target encoding won't work. However, if the target column is accidentally included, it would cause severe data leakage.

### Issue 3: Cross-Validation in Hyperparameter Optimization

In `src/models/trainer.py`, during hyperparameter optimization, the code does:
```python
for train_idx, val_idx in cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Feature engineering is done on X_tr, but target encoding needs y_tr
    # If target encoding uses the entire X_tr including target, it leaks
```

## Why AUC is Close to 1

1. **Target Encoding Leakage**: If target encoding is calculated using the same data being predicted, the model can "see" the answer
2. **Perfect Feature**: Target-encoded features become perfect predictors because they directly encode the target distribution
3. **Overfitting**: The model memorizes the training data patterns instead of learning generalizable patterns

## Solution

We need to implement **cross-validation safe target encoding** (also called "leave-one-out" or "out-of-fold" encoding).

### Fix Strategy

1. **Modify FeatureEngineer to accept target separately**
2. **Implement CV-safe target encoding** that excludes the current sample when calculating means
3. **Update training pipeline** to pass target variable correctly
4. **Add data leakage detection** utilities

## Implementation

The fix will:
- Add `transform_with_target()` method that accepts target separately
- Implement leave-one-out or out-of-fold target encoding
- Update training pipeline to use CV-safe encoding
- Add validation to prevent accidental target inclusion in features
