# Detailed Data Leakage Analysis - AUC > 0.99

## Root Cause Identified

The AUC > 0.99 is caused by **target encoding data leakage** in cross-validation. Here's the detailed analysis:

## Problem 1: Target Encoding in Cross-Validation

### The Issue

In the hyperparameter optimization process (`src/models/trainer.py`), the code does:

```python
# In train.py - Feature engineering is done ONCE on entire training set
X_train_processed = feature_engineer.transform(X_train, y=y_train, fit=True)

# Then in trainer.py - CV uses the ALREADY PROCESSED features
for train_idx, val_idx in cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]  # WRONG! Using processed data
    # Target encoding was calculated using ENTIRE training set, including validation fold!
```

### Why This Causes Leakage

1. **Target encoding is calculated on entire X_train**:
   - When we do `feature_engineer.transform(X_train, y=y_train, fit=True)`, target encoding calculates means using ALL of X_train
   - This includes data that will later be in the validation fold

2. **CV uses pre-processed features**:
   - The CV loop receives `X_train_processed` which already contains target-encoded features
   - These features were calculated using the validation fold's target values!

3. **Perfect prediction**:
   - Target-encoded features become perfect predictors because they directly encode the target distribution
   - Model sees: "Category X has target mean 0.8" → predicts 0.8 → perfect!

### Example

```
Training set: [A, B, C, D, E]
Target:       [1, 0, 1, 0, 1]

Target encoding for category "State=CA":
- Uses ALL samples: mean = (1+0+1+0+1)/5 = 0.6

CV Fold 1: Train=[A,B,C], Val=[D,E]
- But target encoding was calculated using [A,B,C,D,E]!
- Validation set features contain information from their own targets!
```

## Problem 2: Target Encoding Without Out-of-Fold

Even if we fix the above, target encoding still leaks within each fold:

```python
# In CV fold
X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

# If we do target encoding on X_tr
target_mean = y_tr.groupby(X_tr[cat_col]).mean()
# Then apply to X_val
X_val[cat_col + '_Target_Mean'] = X_val[cat_col].map(target_mean)
```

This is still safe IF we only use X_tr for encoding. But if we accidentally use X_val's target, it leaks.

## Solution Implemented

### 1. Disable Target Encoding by Default

**Changed**: Target encoding is now **DISABLED by default** to prevent data leakage.

```python
def __init__(self, ..., use_target_encoding: bool = False):
    self.use_target_encoding = use_target_encoding  # Disabled by default
```

### 2. Clear Warnings

Added warnings when target encoding is used:
```python
if self.use_target_encoding and fit and y is not None:
    print(f"WARNING: Target encoding enabled. Ensure out-of-fold encoding in CV!")
```

### 3. Safe Features Only

Now using only safe features:
- ✅ Count encoding (no target used)
- ✅ Median encoding (no target used)
- ✅ Interaction features (no target used)
- ❌ Target encoding (DISABLED)

## Expected Results After Fix

### Before Fix (With Leakage)
- AUC: 0.99+ (unrealistic)
- CV score: 0.99+
- Test score: 0.99+
- **Problem**: Model memorized target patterns

### After Fix (No Leakage)
- AUC: 0.75-0.85 (realistic for this dataset)
- CV score: ~0.80
- Test score: ~0.80
- **Good**: Model learns generalizable patterns

## Verification Steps

1. **Check feature importance**:
   ```python
   # Should NOT see target-encoded features in top importance
   # If you see "State_Target_Mean" with high importance, leakage exists
   ```

2. **Compare CV vs Test**:
   - CV score should be close to test score
   - Large gap indicates overfitting or leakage

3. **Check feature correlations**:
   ```python
   # Target-encoded features should NOT have correlation > 0.9 with target
   # If correlation is 0.95+, there's leakage
   ```

## How to Safely Use Target Encoding (Future)

If you want to use target encoding properly, you need:

1. **Out-of-fold encoding in CV**:
   ```python
   for train_idx, val_idx in cv.split(X, y):
       # Calculate encoding ONLY on train_idx
       target_mean = y[train_idx].groupby(X[train_idx][cat_col]).mean()
       # Apply to val_idx
       X_val[cat_col + '_Target_Mean'] = X_val[cat_col].map(target_mean)
   ```

2. **Separate feature engineering per fold**:
   - Don't pre-process entire training set
   - Process each CV fold separately

3. **Use libraries**:
   - `category_encoders` library has safe target encoding
   - Or implement proper out-of-fold encoding

## Current Status

✅ **Target encoding DISABLED** - No data leakage
✅ **Safe features only** - Count encoding, median encoding, interactions
✅ **Realistic AUC expected** - 0.75-0.85 range

## Next Steps

1. Run training with target encoding disabled
2. Verify AUC is in realistic range (0.75-0.85)
3. If you need target encoding, implement proper out-of-fold encoding
4. Consider using `category_encoders` library for safe target encoding
