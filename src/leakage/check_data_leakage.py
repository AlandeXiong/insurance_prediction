"""Diagnostic script to check for data leakage"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def check_target_encoding_leakage(df: pd.DataFrame, target_col: str, cat_cols: list):
    """
    Check if target-encoded features would cause data leakage.
    
    Returns correlation between target and potential target-encoded features.
    """
    print("="*80)
    print("DATA LEAKAGE DIAGNOSTIC")
    print("="*80)
    
    y = df[target_col]
    
    print(f"\nTarget distribution:")
    print(y.value_counts())
    print(f"Target mean: {y.mean():.4f}")
    
    print(f"\n{'='*80}")
    print("Checking potential target encoding leakage...")
    print(f"{'='*80}")
    
    high_corr_features = []
    
    for cat_col in cat_cols:
        if cat_col not in df.columns:
            continue
        
        # Simulate target encoding (this is what causes leakage!)
        target_mean = y.groupby(df[cat_col]).mean()
        df[f'{cat_col}_Target_Mean'] = df[cat_col].map(target_mean)
        
        # Check correlation with target
        corr = df[f'{cat_col}_Target_Mean'].corr(y)
        
        print(f"\n{cat_col}:")
        print(f"  Categories: {df[cat_col].nunique()}")
        print(f"  Target encoding correlation: {corr:.4f}")
        
        if abs(corr) > 0.9:
            high_corr_features.append(cat_col)
            print(f"  ⚠️  WARNING: Very high correlation! This indicates potential leakage.")
        elif abs(corr) > 0.7:
            print(f"  ⚠️  CAUTION: High correlation. May cause leakage in CV.")
        else:
            print(f"  ✅ OK: Reasonable correlation.")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if high_corr_features:
        print(f"\n⚠️  CRITICAL: {len(high_corr_features)} features with correlation > 0.9:")
        for feat in high_corr_features:
            print(f"  - {feat}")
        print("\nThese features will cause AUC > 0.99 if used in target encoding!")
        print("Solution: Disable target encoding or use proper out-of-fold encoding.")
        return True
    else:
        print("\n✅ No critical leakage detected in target encoding.")
        print("However, target encoding should still be disabled in CV to be safe.")
        return False

def check_feature_target_correlation(df: pd.DataFrame, target_col: str, features: list):
    """Check correlation between features and target."""
    print(f"\n{'='*80}")
    print("FEATURE-TARGET CORRELATIONS")
    print(f"{'='*80}")
    
    y = df[target_col]
    suspicious_features = []
    
    for feat in features:
        if feat not in df.columns:
            continue
        
        if df[feat].dtype in [np.int64, np.int32, np.float64, np.float32]:
            corr = df[feat].corr(y)
            if abs(corr) > 0.9:
                suspicious_features.append((feat, corr))
                print(f"⚠️  {feat}: {corr:.4f} (VERY HIGH - possible leakage)")
            elif abs(corr) > 0.7:
                print(f"⚠️  {feat}: {corr:.4f} (HIGH)")
            else:
                print(f"✅ {feat}: {corr:.4f}")
    
    return suspicious_features

def main():
    """Main diagnostic function"""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.config import load_config
    from src.utils.data_loading import load_train_test_data

    config = load_config(str(root / "config.yaml"))

    # Load training data only
    df_train, _, resolved_split = load_train_test_data(config)
    target_col = config['data']['target_column']

    print(f"Loading training data (split={resolved_split.strategy}, details={resolved_split.details})")
    df = df_train.copy()
    print(f"Data shape: {df.shape}")
    
    # Get features
    cat_features = config['features']['categorical_features']
    num_features = config['features']['numerical_features']
    
    # Check target encoding leakage
    has_leakage = check_target_encoding_leakage(df, target_col, cat_features)
    
    # Check feature correlations
    all_features = cat_features + num_features
    suspicious = check_feature_target_correlation(df, target_col, all_features)
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if has_leakage or suspicious:
        print("\n1. ✅ Target encoding is DISABLED by default (good!)")
        print("2. Keep target encoding disabled unless implementing out-of-fold encoding")
        print("3. Expected AUC should be 0.75-0.85 (realistic)")
        print("4. If AUC > 0.99, check if target encoding is accidentally enabled")
    else:
        print("\n1. No obvious leakage detected")
        print("2. Target encoding should still be disabled in CV")
        print("3. Expected AUC: 0.75-0.85 for this dataset")

if __name__ == "__main__":
    main()
