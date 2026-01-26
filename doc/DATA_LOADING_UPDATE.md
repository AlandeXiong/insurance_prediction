# Data Loading Update - Separate Train/Test Files

## Overview

The system now supports loading separate training and test files, in addition to the original single-file with split approach.

## Configuration

### Option 1: Separate Train/Test Files (Recommended)

```yaml
data:
  train_path: "dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis—train.csv"
  test_path: "dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis—test.csv"
  use_separate_files: true
  target_column: "Response"
```

### Option 2: Single File with Split

```yaml
data:
  train_path: "dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv"
  use_separate_files: false
  target_column: "Response"
  test_size: 0.2
  random_state: 42
  stratify: true
```

## Changes Made

### 1. config.yaml
- Added `test_path` for test file location
- Added `use_separate_files` flag to control behavior
- `test_size`, `random_state`, `stratify` only used when `use_separate_files: false`

### 2. train.py
- Modified data loading to support both modes
- When `use_separate_files: true`:
  - Loads train and test files separately
  - Uses training data for EDA and feature discovery
  - Fits target encoder on training data only
  - Transforms test data using fitted encoder
  - Handles unseen labels in test set gracefully
- When `use_separate_files: false`:
  - Uses original logic: load single file and split

## Key Features

1. **Target Encoding Safety**:
   - Target encoder is fitted ONLY on training data
   - Test data is transformed using the fitted encoder
   - Unseen labels in test set are handled with warnings

2. **EDA on Training Data Only**:
   - EDA is performed on training data to prevent information leakage
   - Feature discovery uses training data only

3. **Backward Compatibility**:
   - Original single-file approach still works
   - Set `use_separate_files: false` to use original behavior

## File Structure

```
dataset/
├── WA_Fn-UseC_-Marketing-Customer-Value-Analysis—train.csv  # Training set
├── WA_Fn-UseC_-Marketing-Customer-Value-Analysis—test.csv    # Test set
└── WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv         # Original (optional)
```

## Usage

1. **With Separate Files** (Current Setup):
   ```yaml
   use_separate_files: true
   train_path: "dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis—train.csv"
   test_path: "dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis—test.csv"
   ```

2. **With Single File**:
   ```yaml
   use_separate_files: false
   train_path: "dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv"
   test_size: 0.2
   ```

## Benefits

- ✅ **Reproducibility**: Fixed train/test split ensures consistent results
- ✅ **No Data Leakage**: Test data never used for training or feature engineering
- ✅ **Flexibility**: Supports both separate files and single file approaches
- ✅ **Safety**: Handles edge cases like unseen labels in test set
