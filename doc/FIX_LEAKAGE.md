# Complete Data Leakage Fix Guide

## Problem: AUC Still > 0.99

If you see AUC > 0.99, it usually indicates **data leakage** or a train/test contamination issue.

⚠️ Note: **Target encoding is no longer used in the current pipeline.** It was removed because it is
easy to leak without strict out-of-fold encoding. The pipeline uses leakage-safe alternatives:
count/frequency encodings and group aggregations (median/mean/std) on numeric features.

## Root Causes

### 1. Old Saved Models / Feature Engineer Artifacts

**Problem**: Old saved `feature_engineer.pkl` / model artifacts may not match the current codebase,
and can cause confusing behavior.

**Solution**:
```bash
# Delete old models to start fresh
rm -rf models/*.pkl
rm -rf models/feature_engineer.pkl
```

### 3. Features Directly Containing Target Information

**Check**: Run diagnostic script
```bash
python diagnose_leakage.py
```

**Common culprits**:
- Features with correlation > 0.9 to target
- Features that are perfect predictors
- Duplicate rows with different targets

### 4. Data Issues

**Check for**:
- Duplicate rows
- Perfect separability (one feature perfectly predicts target)
- Leakage from future information

## Complete Fix Steps

### Step 1: Clean Old Models

```bash
cd /Users/xiongjian/PycharmProjects/InsurancePrediction
rm -rf models/*.pkl
rm -rf models/feature_engineer.pkl
```

### Step 2: Run Diagnostic

```bash
python diagnose_leakage.py
```

This will check:
- Target column in features
- High correlation features
- Existing target encoding maps
- Configuration settings
- Data quality issues

### Step 3: Re-train from Scratch

```bash
python train.py
```

### Step 4: Verify Results

After training, check:
- AUC should be 0.75-0.85 (realistic)
- CV score ≈ test score (no large gap)
- Feature importance doesn't show suspiciously-perfect predictors

## Code Changes Made

### 1. Feature Engineering (`src/features/engineering.py`)

**Change**: Target encoding logic was removed from the pipeline to eliminate leakage risk.
Legacy keys may still exist in saved artifacts for backward compatibility but are not used.

## Expected Results

### Before Fix
- AUC: 0.99+ (unrealistic)
- CV: 0.99+
- Test: 0.99+
- **Problem**: Data leakage

### After Fix
- AUC: 0.75-0.85 (realistic for this dataset)
- CV: ~0.80
- Test: ~0.80
- **Good**: No leakage, realistic performance

## Verification Checklist

- [ ] Deleted old models (`models/*.pkl`)
- [ ] Ran diagnostic script (no critical issues)
- [ ] Re-trained from scratch
- [ ] AUC is 0.75-0.85 (not 0.99+)
- [ ] CV score ≈ test score
- [ ] No suspiciously-perfect predictors in top importance

## If AUC Still > 0.99

1. **Check feature importance**:
   ```python
   # Should NOT see suspiciously-perfect predictors
   ```

2. **Check correlations**:
   ```python
   # Run: python diagnose_leakage.py
   # Check for features with correlation > 0.9
   ```

3. **Check data**:
   - Look for duplicate rows
   - Check for perfect separability
   - Verify no future information leakage

4. **Check logs**:
   - CV metrics should be close to test metrics (no huge gap)

## Additional Notes

- Target encoding is **not used** in the current pipeline
- Old saved models may still contain legacy keys - delete them before re-training
- Diagnostic script helps identify all leakage sources
- Realistic AUC for this dataset: 0.75-0.85
