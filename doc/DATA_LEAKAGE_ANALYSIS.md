# Data Leakage Analysis and Fix

## Problem Identified

Historically, the code had a **critical data leakage issue** in target encoding that could cause AUC
to be artificially high (close to 1).

✅ Current status: target encoding is **not used** in the current pipeline.

## Root Cause

### Issue 1 (historical): Target Encoding Without Proper Cross-Validation

In older versions, `create_statistical_features` performed target encoding:

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

## Solution (current approach)

We do NOT use target encoding. Instead we:
- fit feature engineering per fold for CV/OOF routines
- use leakage-safe encodings (count/frequency + group aggregations on numeric features)

### Fix Strategy

1. Fit feature engineering per fold during CV/OOF
2. Use leakage-safe statistical features (no target)
3. Keep explicit separation between features and target
4. Keep leakage detection utilities

## Implementation

The fix will:
- Add `transform_with_target()` method that accepts target separately
- Implement leave-one-out or out-of-fold target encoding
- Update training pipeline to use CV-safe encoding
- Add validation to prevent accidental target inclusion in features
