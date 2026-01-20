"""Comprehensive diagnostic script to find data leakage sources"""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys

def check_all_leakage_sources():
    """Check all possible sources of data leakage"""
    print("="*80)
    print("COMPREHENSIVE DATA LEAKAGE DIAGNOSTIC")
    print("="*80)
    
    # Load config
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("Error: config.yaml not found")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    data_path = config['data']['train_path']
    target_col = config['data']['target_column']
    
    print(f"\n1. Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Data shape: {df.shape}")
    print(f"   Target column: {target_col}")
    print(f"   Target distribution:\n{df[target_col].value_counts()}")
    
    # Check 1: Target column in features
    print(f"\n{'='*80}")
    print("CHECK 1: Target column accidentally in features?")
    print(f"{'='*80}")
    cat_features = config['features']['categorical_features']
    num_features = config['features']['numerical_features']
    all_features = cat_features + num_features
    
    if target_col in all_features:
        print(f"❌ CRITICAL: Target column '{target_col}' is in feature list!")
        return True
    else:
        print(f"✅ OK: Target column not in feature list")
    
    # Check 2: Features with perfect correlation
    print(f"\n{'='*80}")
    print("CHECK 2: Features with very high correlation to target")
    print(f"{'='*80}")
    
    y = (df[target_col] == 'Yes').astype(int) if df[target_col].dtype == 'object' else df[target_col]
    
    high_corr_features = []
    for feat in all_features:
        if feat not in df.columns:
            continue
        if df[feat].dtype in [np.int64, np.int32, np.float64, np.float32]:
            corr = abs(df[feat].corr(y))
            if corr > 0.9:
                high_corr_features.append((feat, corr))
                print(f"❌ {feat}: correlation = {corr:.4f} (VERY HIGH - possible leakage)")
            elif corr > 0.7:
                print(f"⚠️  {feat}: correlation = {corr:.4f} (HIGH)")
    
    if high_corr_features:
        print(f"\n⚠️  Found {len(high_corr_features)} features with correlation > 0.9")
        return True
    
    # Check 3: Target encoding maps exist
    print(f"\n{'='*80}")
    print("CHECK 3: Existing target encoding maps (from previous runs)")
    print(f"{'='*80}")
    
    models_dir = Path(config['paths']['models_dir'])
    feature_engineer_path = models_dir / 'feature_engineer.pkl'
    
    if feature_engineer_path.exists():
        import joblib
        try:
            data = joblib.load(feature_engineer_path)
            target_maps = data.get('target_encoding_maps', {})
            use_target_encoding = data.get('use_target_encoding', False)
            
            if target_maps:
                print(f"⚠️  Found {len(target_maps)} target encoding maps in saved feature engineer")
                print(f"   use_target_encoding setting: {use_target_encoding}")
                if use_target_encoding:
                    print(f"❌ CRITICAL: Target encoding is ENABLED in saved model!")
                    return True
                else:
                    print(f"   Note: Maps exist but encoding is disabled (should be safe)")
            else:
                print(f"✅ No target encoding maps found")
        except Exception as e:
            print(f"⚠️  Could not load feature engineer: {e}")
    else:
        print(f"✅ No saved feature engineer found (first run)")
    
    # Check 4: Simulate target encoding leakage
    print(f"\n{'='*80}")
    print("CHECK 4: Simulating target encoding (to check leakage potential)")
    print(f"{'='*80}")
    
    max_corr = 0
    worst_feature = None
    for cat_col in cat_features:
        if cat_col not in df.columns:
            continue
        
        # Simulate target encoding
        target_mean = y.groupby(df[cat_col]).mean()
        encoded = df[cat_col].map(target_mean)
        corr = abs(encoded.corr(y))
        
        if corr > max_corr:
            max_corr = corr
            worst_feature = cat_col
        
        if corr > 0.9:
            print(f"❌ {cat_col}_Target_Mean would have correlation: {corr:.4f} (CRITICAL)")
        elif corr > 0.7:
            print(f"⚠️  {cat_col}_Target_Mean would have correlation: {corr:.4f} (HIGH)")
    
    if max_corr > 0.9:
        print(f"\n❌ CRITICAL: Target encoding would cause severe leakage!")
        print(f"   Worst feature: {worst_feature} (correlation: {max_corr:.4f})")
        return True
    
    # Check 5: Duplicate or near-duplicate rows
    print(f"\n{'='*80}")
    print("CHECK 5: Duplicate or highly similar rows")
    print(f"{'='*80}")
    
    # Check exact duplicates
    duplicates = df.duplicated(subset=[f for f in all_features if f in df.columns]).sum()
    if duplicates > 0:
        print(f"⚠️  Found {duplicates} duplicate rows (same features, possibly different targets)")
        if duplicates > len(df) * 0.1:
            print(f"❌ CRITICAL: Too many duplicates ({duplicates}/{len(df)} = {duplicates/len(df)*100:.1f}%)")
            return True
    
    # Check 6: Config check
    print(f"\n{'='*80}")
    print("CHECK 6: Configuration settings")
    print(f"{'='*80}")
    
    use_target_encoding = config.get('features', {}).get('use_target_encoding', False)
    print(f"use_target_encoding in config: {use_target_encoding}")
    
    if use_target_encoding:
        print(f"❌ CRITICAL: Target encoding is ENABLED in config.yaml!")
        print(f"   This will cause AUC > 0.99. Set use_target_encoding: false")
        return True
    else:
        print(f"✅ Target encoding is DISABLED in config")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("✅ No critical leakage sources detected in current configuration")
    print("\nIf AUC is still > 0.99, possible causes:")
    print("1. Old feature engineer with target encoding enabled - delete models/ folder")
    print("2. Features accidentally contain target information")
    print("3. Data issue (duplicates, perfect separability)")
    print("4. Model overfitting (check CV vs test score gap)")
    
    return False

if __name__ == "__main__":
    has_leakage = check_all_leakage_sources()
    sys.exit(1 if has_leakage else 0)
