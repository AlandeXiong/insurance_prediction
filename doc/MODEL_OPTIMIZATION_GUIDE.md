# Model Optimization Guide - Class Imbalance & Threshold Tuning

## Problem Analysis

Current model performance shows:
- **Low Recall** (0.06-0.09): Models are too conservative, missing most positive cases
- **High Precision** (0.79-0.96): But this is because models rarely predict positive class
- **Low F1 Score** (0.12-0.17): Overall poor performance

This is a classic **class imbalance problem**.

## Solution Implemented

### 1. Unified Feature Engineering Pipeline

**All models use the same pipeline:**
- ✅ Same feature engineering (`FeatureEngineer.transform()`)
- ✅ Same preprocessing steps
- ✅ Same feature order
- ✅ Applied consistently to train, validation, and test sets

### 2. Class Imbalance Handling

**Automatic class weight computation:**
- Computes balanced class weights from training data
- Applied to all models:
  - **XGBoost**: `scale_pos_weight` parameter
  - **CatBoost**: `class_weights` parameter
  - **Ensemble**: Inherits from base models

### 3. Threshold Optimization

**Optimal threshold finding:**
- Finds threshold that meets minimum recall requirement (≥0.6)
- Among valid thresholds, selects one with highest precision
- Applied during:
  - Hyperparameter optimization (CV folds)
  - Final model training (validation set)
  - Test evaluation uses the **validation-selected** threshold (no test-label leakage)

### 4. Optimization Metric Change

**From AUC/F1 to Precision@Recall Constraint:**
- **Before**: Optimized for ROC-AUC / F1 (can still miss the recall constraint)
- **After**: Optimizes **precision subject to recall ≥ min_recall** (matches the business goal)
- This is the standard Kaggle-style approach for imbalanced problems: train good probabilities, then do **threshold moving**

## Configuration

```yaml
model:
  scoring: "precision_at_min_recall"
  min_recall: 0.6  # Minimum recall requirement
  optimize_threshold: true  # Enable threshold optimization
```

## Training Pipeline

### Step 1: Feature Engineering (Unified)
```python
# All models use the same feature engineer
X_train_processed = feature_engineer.transform(X_train, y=y_train, fit=True)
X_test_processed = feature_engineer.transform(X_test, fit=False)
```

### Step 2: Create Validation Set
```python
# Split training data for threshold optimization
X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(
    X_train_processed, y_train, test_size=0.2, stratify=y_train
)
```

### Step 3: Model Training (Unified Pipeline)
All models:
1. Compute class weights from training data
2. Train with class weights
3. Optimize threshold on validation set
4. Store optimal threshold

### Step 4: Evaluation
- Uses optimal thresholds (not default 0.5)
- Reports metrics at optimal threshold
- Also reports metrics at default threshold for comparison

## Expected Improvements

### Before Optimization
- Recall: 0.06-0.09 (too low)
- Precision: 0.79-0.96 (artificially high)
- F1: 0.12-0.17 (poor)

### After Optimization
- Recall: ≥0.6 (meets requirement)
- Precision: Optimized while maintaining recall
- F1: Should improve significantly (target: 0.5+)

## Key Features

1. **Unified Pipeline**: All models use identical feature engineering
2. **Class Weights**: Automatic handling of class imbalance
3. **Threshold Optimization**: Meets recall requirement, maximizes precision
4. **F1 Optimization**: Balances precision and recall during training
5. **Consistent Evaluation**: Same metrics and thresholds for all models

## Verification

After training, check:
- ✅ All models show recall ≥ 0.6
- ✅ Precision is optimized (should be reasonable, not 0.99)
- ✅ F1 score improved significantly
- ✅ All models use same feature engineering
- ✅ Thresholds are optimized and saved
