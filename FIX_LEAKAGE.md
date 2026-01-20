# Complete Data Leakage Fix Guide

## Problem: AUC Still > 0.99

Even after disabling target encoding, AUC may still be > 0.99. Here are all possible causes and fixes:

## Root Causes

### 1. Old Saved Models with Target Encoding

**Problem**: Previously saved feature engineer may have target encoding enabled.

**Solution**:
```bash
# Delete old models to start fresh
rm -rf models/*.pkl
rm -rf models/feature_engineer.pkl
```

### 2. Target Encoding Maps Still Being Used

**Problem**: Even with `use_target_encoding=False`, old maps might be loaded and used.

**Fix Applied**: 
- Modified `create_statistical_features()` to only use target encoding maps if `use_target_encoding=True`
- Clear target encoding maps when disabled

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

### Step 2: Verify Config

Check `config.yaml`:
```yaml
features:
  use_target_encoding: false  # MUST be false
```

### Step 3: Run Diagnostic

```bash
python diagnose_leakage.py
```

This will check:
- Target column in features
- High correlation features
- Existing target encoding maps
- Configuration settings
- Data quality issues

### Step 4: Re-train from Scratch

```bash
python train.py
```

### Step 5: Verify Results

After training, check:
- AUC should be 0.75-0.85 (realistic)
- CV score ≈ test score (no large gap)
- Feature importance doesn't show target-encoded features

## Code Changes Made

### 1. Feature Engineering (`src/features/engineering.py`)

**Change**: Only use target encoding maps if explicitly enabled
```python
# Before: Would use maps even if disabled
elif hasattr(self, 'target_encoding_maps') and cat_col in self.target_encoding_maps:

# After: Only use if enabled
elif self.use_target_encoding and hasattr(self, 'target_encoding_maps') and cat_col in self.target_encoding_maps:
```

### 2. Clear Maps When Disabled

**Change**: Clear target encoding maps when disabled
```python
if not self.use_target_encoding:
    self.target_encoding_maps = {}  # Keep empty
```

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
- [ ] Config has `use_target_encoding: false`
- [ ] Ran diagnostic script (no critical issues)
- [ ] Re-trained from scratch
- [ ] AUC is 0.75-0.85 (not 0.99+)
- [ ] CV score ≈ test score
- [ ] No target-encoded features in top importance

## If AUC Still > 0.99

1. **Check feature importance**:
   ```python
   # Should NOT see features ending with _Target_Mean
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
   - Should see "Target encoding is DISABLED"
   - Should NOT see "WARNING: Target encoding enabled"

## Additional Notes

- Target encoding is **completely disabled** by default
- Old saved models may still have target encoding - delete them
- Diagnostic script helps identify all leakage sources
- Realistic AUC for this dataset: 0.75-0.85
