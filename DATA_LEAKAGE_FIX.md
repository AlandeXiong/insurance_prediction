# Data Leakage Fix Documentation

## Problem Identified

The original implementation had a **critical data leakage risk** in target encoding:

1. **Issue**: The `transform()` method was checking if target column exists in the dataframe (`if self.target_column in df.columns`), but during training, `X_train` doesn't contain the target column.

2. **Risk**: If target encoding was somehow using test data or future information, it would cause:
   - Unrealistically high AUC scores (close to 1.0)
   - Poor generalization to new data
   - Overfitting

## Solution Implemented

### 1. Separated Target Variable from Features

**Before (RISKY):**
```python
# Target encoding checked if target was in dataframe
if fit and self.target_column in df.columns:
    target_mean = df.groupby(cat_col)[self.target_column].mean()
```

**After (SAFE):**
```python
# Target encoding now requires y to be passed separately
def transform(self, df: pd.DataFrame, y: pd.Series = None, fit: bool = True):
    # y is passed separately, ensuring only training data is used
    if fit and y is not None:
        target_mean = pd.Series(y.values, index=df.index).groupby(df[cat_col]).mean()
```

### 2. Safety Checks Added

- **Automatic target removal**: If target column is accidentally included in features, it's automatically removed
- **Warning messages**: Clear warnings when target encoding is skipped due to missing y
- **Explicit separation**: Target must be passed as separate parameter, making data flow explicit

### 3. Training Pipeline Updated

**Before:**
```python
X_train_processed = feature_engineer.transform(X_train, fit=True)
```

**After:**
```python
# Explicitly pass y_train separately to prevent leakage
X_train_processed = feature_engineer.transform(X_train, y=y_train, fit=True)
# Test data uses stored mappings (no y needed)
X_test_processed = feature_engineer.transform(X_test, fit=False)
```

## Key Changes

### `src/features/engineering.py`

1. **`create_statistical_features()` method**:
   - Now accepts `y: pd.Series` parameter
   - Only uses `y` when `fit=True` (training)
   - Uses stored mappings when `fit=False` (prediction)
   - Added safety warnings

2. **`transform()` method**:
   - Now accepts `y: pd.Series` parameter
   - Automatically removes target column if accidentally included
   - Explicitly separates feature data from target data

### `train.py`

- Updated to pass `y_train` separately during feature engineering
- Added logging to indicate data leakage prevention measures

### `src/api/predictor.py`

- Updated to use new transform signature (y=None for prediction)

## Verification

To verify the fix works correctly:

1. **Check training logs**: Should see "Target encoding uses only training data to prevent data leakage"
2. **AUC scores**: Should be realistic (typically 0.7-0.9 for this dataset, not 0.99+)
3. **Cross-validation**: CV scores should be close to test scores (no large gap)
4. **Feature importance**: Target-encoded features should have reasonable importance

## Best Practices Followed

1. ✅ **Train/Test Separation**: Target encoding only uses training data
2. ✅ **Explicit Parameters**: Target passed separately, making data flow clear
3. ✅ **Safety Checks**: Automatic detection and removal of target from features
4. ✅ **Documentation**: Clear warnings and comments about data leakage prevention

## Testing Recommendations

After this fix, you should see:
- More realistic AUC scores (0.75-0.85 range is typical)
- Better generalization (CV score ≈ test score)
- No suspiciously perfect predictions

If AUC is still very high (>0.95), check:
- Are there other features that might leak information?
- Is the dataset too small or too easy?
- Are there duplicate or near-duplicate samples?
