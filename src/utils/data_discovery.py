"""Data and feature discovery utilities following Kaggle best practices"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any


def discover_features(df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
    """
    Automatically discover categorical and numerical features from dataset.
    Following Kaggle best practices for feature identification.
    
    Args:
        df: Input dataframe
        target_column: Name of target column (if exists)
    
    Returns:
        Dictionary with discovered features and metadata
    """
    # Exclude target and ID columns
    exclude_cols = [target_column] if target_column and target_column in df.columns else []
    exclude_cols.extend(['Customer', 'ID', 'id', 'Id'])
    
    all_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Discover numerical features
    numerical_features = []
    for col in all_cols:
        if df[col].dtype in [np.int64, np.int32, np.float64, np.float32]:
            # Check if it's actually categorical (low cardinality integer)
            if df[col].dtype in [np.int64, np.int32]:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() < 20:
                    # Likely categorical encoded as integer
                    continue
            numerical_features.append(col)
    
    # Discover categorical features
    categorical_features = []
    for col in all_cols:
        if col not in numerical_features:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_features.append(col)
            elif df[col].dtype in [np.int64, np.int32]:
                # Low cardinality integer might be categorical
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() < 20:
                    categorical_features.append(col)
    
    # Discover date features
    date_features = []
    for col in all_cols:
        if df[col].dtype == 'object':
            # Try to parse as date
            try:
                pd.to_datetime(df[col].dropna().iloc[0])
                date_features.append(col)
            except:
                pass
    
    # Calculate feature statistics
    feature_stats = {
        'numerical': {
            'count': len(numerical_features),
            'features': numerical_features,
            'missing_ratios': {col: df[col].isnull().sum() / len(df) 
                              for col in numerical_features}
        },
        'categorical': {
            'count': len(categorical_features),
            'features': categorical_features,
            'cardinalities': {col: df[col].nunique() 
                            for col in categorical_features},
            'missing_ratios': {col: df[col].isnull().sum() / len(df) 
                             for col in categorical_features}
        },
        'date': {
            'count': len(date_features),
            'features': date_features
        }
    }
    
    return {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'date_features': date_features,
        'statistics': feature_stats,
        'total_features': len(numerical_features) + len(categorical_features)
    }


def analyze_feature_importance(feature_importance_dict: Dict[str, float], 
                               top_n: int = 20) -> pd.DataFrame:
    """
    Analyze and rank feature importance.
    
    Args:
        feature_importance_dict: Dictionary of feature names and importance scores
        top_n: Number of top features to return
    
    Returns:
        DataFrame with ranked features
    """
    df = pd.DataFrame([
        {'feature': k, 'importance': v} 
        for k, v in feature_importance_dict.items()
    ])
    df = df.sort_values('importance', ascending=False)
    df['rank'] = range(1, len(df) + 1)
    df['cumulative_importance'] = df['importance'].cumsum() / df['importance'].sum()
    
    return df.head(top_n)


def detect_feature_interactions(df: pd.DataFrame, 
                               numerical_features: List[str],
                               target_column: str = None,
                               top_k: int = 10) -> List[Tuple[str, str, float]]:
    """
    Detect potential feature interactions using correlation analysis.
    
    Args:
        df: Input dataframe
        numerical_features: List of numerical feature names
        target_column: Target column name (if exists)
        top_k: Number of top interactions to return
    
    Returns:
        List of tuples (feature1, feature2, correlation_score)
    """
    interactions = []
    
    if target_column and target_column in df.columns:
        # Correlations with target
        for feat in numerical_features:
            if feat in df.columns:
                corr = df[[feat, target_column]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    interactions.append((feat, target_column, abs(corr)))
    
    # Feature-feature correlations
    num_df = df[numerical_features].select_dtypes(include=[np.number])
    corr_matrix = num_df.corr().abs()
    
    # Get top correlations (excluding diagonal)
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            feat1 = corr_matrix.columns[i]
            feat2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val) and corr_val > 0.3:  # Threshold for meaningful correlation
                interactions.append((feat1, feat2, corr_val))
    
    # Sort by correlation strength
    interactions.sort(key=lambda x: x[2], reverse=True)
    
    return interactions[:top_k]
