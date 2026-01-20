"""Advanced feature engineering following Kaggle best practices"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Any, Optional
import joblib
from pathlib import Path


class FeatureEngineer:
    """
    Advanced feature engineering for insurance renewal prediction.
    Follows Kaggle competition best practices for feature creation and transformation.
    
    IMPORTANT: Target encoding is disabled by default to prevent data leakage.
    Enable it only if you implement proper out-of-fold encoding in cross-validation.
    """
    
    def __init__(self, categorical_features: List[str], 
                 numerical_features: List[str],
                 target_column: str = "Response",
                 use_target_encoding: bool = False):
        """
        Initialize feature engineer.
        
        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            target_column: Name of target column
            use_target_encoding: Whether to use target encoding (DISABLED by default to prevent leakage)
        """
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target_column = target_column
        self.use_target_encoding = use_target_encoding  # Disabled by default
        
        # Encoders and transformers
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_order = None  # Store feature order for prediction consistency
        
        # Statistical encoding maps
        self.target_encoding_maps = {}
        self.count_encoding_maps = {}
        self.median_encoding_maps = {}
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction and derived features.
        Following Kaggle best practices for feature engineering.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dataframe with new interaction features
        """
        df = df.copy()
        
        # Premium efficiency features
        if 'Monthly Premium Auto' in df.columns and 'Number of Policies' in df.columns:
            df['Premium_Per_Policy'] = df['Monthly Premium Auto'] / (df['Number of Policies'] + 1)
            df['Premium_Per_Policy_Log'] = np.log1p(df['Premium_Per_Policy'])
        
        # Claim-related features
        if 'Number of Open Complaints' in df.columns and 'Months Since Policy Inception' in df.columns:
            df['Claim_Frequency'] = df['Number of Open Complaints'] / (df['Months Since Policy Inception'] + 1)
            df['Complaints_Per_Year'] = df['Number of Open Complaints'] / ((df['Months Since Policy Inception'] / 12) + 1)
        
        # Financial features
        if 'Total Claim Amount' in df.columns and 'Number of Policies' in df.columns:
            df['Avg_Claim_Amount'] = df['Total Claim Amount'] / (df['Number of Policies'] + 1)
            df['Claim_Amount_Per_Month'] = df['Total Claim Amount'] / (df['Months Since Policy Inception'] + 1)
        
        # Customer value features
        if 'Customer Lifetime Value' in df.columns and 'Income' in df.columns:
            df['CLV_Income_Ratio'] = df['Customer Lifetime Value'] / (df['Income'] + 1)
            df['CLV_Log'] = np.log1p(df['Customer Lifetime Value'])
            df['Income_Log'] = np.log1p(df['Income'])
        
        # Premium value ratio
        if 'Monthly Premium Auto' in df.columns and 'Income' in df.columns:
            df['Premium_Income_Ratio'] = df['Monthly Premium Auto'] / (df['Income'] + 1)
        
        # Time-based features
        if 'Months Since Last Claim' in df.columns:
            df['Has_Recent_Claim'] = (df['Months Since Last Claim'] <= 12).astype(int)
            df['Has_Very_Recent_Claim'] = (df['Months Since Last Claim'] <= 6).astype(int)
            df['Claim_Recency'] = 1 / (df['Months Since Last Claim'] + 1)
            df['Claim_Recency_Log'] = np.log1p(df['Months Since Last Claim'])
        
        if 'Months Since Policy Inception' in df.columns:
            df['Policy_Age_Years'] = df['Months Since Policy Inception'] / 12
            df['Is_Long_Term_Customer'] = (df['Months Since Policy Inception'] >= 24).astype(int)
            df['Is_Very_Long_Term'] = (df['Months Since Policy Inception'] >= 48).astype(int)
            df['Policy_Age_Log'] = np.log1p(df['Months Since Policy Inception'])
        
        # Risk indicators
        if 'Number of Open Complaints' in df.columns:
            df['Has_Complaints'] = (df['Number of Open Complaints'] > 0).astype(int)
            df['Multiple_Complaints'] = (df['Number of Open Complaints'] > 1).astype(int)
        
        # Policy diversity
        if 'Number of Policies' in df.columns:
            df['Is_Multi_Policy'] = (df['Number of Policies'] > 1).astype(int)
            df['Is_High_Policy_Count'] = (df['Number of Policies'] >= 3).astype(int)
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame, y: Optional[pd.Series] = None, fit: bool = True) -> pd.DataFrame:
        """
        Create statistical aggregation features using count encoding and median encoding.
        
        WARNING: Target encoding is DISABLED by default to prevent data leakage.
        Target encoding can cause severe data leakage in cross-validation if not done properly
        (using out-of-fold encoding). It's disabled here to ensure safe feature engineering.
        
        Args:
            df: Input dataframe (features only, no target column)
            y: Target series (NOT USED - target encoding disabled)
            fit: Whether to fit encoders or use existing mappings
        
        Returns:
            Dataframe with statistical features
        """
        df = df.copy()
        
        # Initialize encoding maps if fitting
        if fit and not hasattr(self, 'target_encoding_maps'):
            self.target_encoding_maps = {}
            self.count_encoding_maps = {}
            self.median_encoding_maps = {}
        
        # Create statistical features for categorical columns
        for cat_col in self.categorical_features:
            if cat_col not in df.columns:
                continue
            
            # TARGET ENCODING DISABLED - This was causing data leakage!
            # Target encoding requires out-of-fold encoding in CV to be safe.
            # For now, we disable it to prevent unrealistic AUC scores.
            if self.use_target_encoding and fit and y is not None:
                # Only enable if explicitly requested AND out-of-fold encoding is implemented
                print(f"WARNING: Target encoding enabled for {cat_col}. Ensure out-of-fold encoding in CV!")
                target_mean = pd.Series(y.values, index=df.index).groupby(df[cat_col]).mean()
                self.target_encoding_maps[cat_col] = target_mean.to_dict()
                df[f'{cat_col}_Target_Mean'] = df[cat_col].map(target_mean)
            elif hasattr(self, 'target_encoding_maps') and cat_col in self.target_encoding_maps:
                # Use stored mapping for prediction
                df[f'{cat_col}_Target_Mean'] = df[cat_col].map(
                    self.target_encoding_maps[cat_col]
                ).fillna(0)
            
            # Count encoding (SAFE - no target information used)
            if fit:
                count_map = df[cat_col].value_counts().to_dict()
                self.count_encoding_maps[cat_col] = count_map
                df[f'{cat_col}_Count'] = df[cat_col].map(count_map)
            elif hasattr(self, 'count_encoding_maps') and cat_col in self.count_encoding_maps:
                df[f'{cat_col}_Count'] = df[cat_col].map(
                    self.count_encoding_maps[cat_col]
                ).fillna(0)
            
            # Median encoding for numerical features grouped by category (SAFE - no target used)
            if fit:
                for num_col in self.numerical_features:
                    if num_col in df.columns:
                        median_map = df.groupby(cat_col)[num_col].median().to_dict()
                        key = f'{cat_col}_{num_col}_Median'
                        self.median_encoding_maps[key] = median_map
                        df[key] = df[cat_col].map(median_map)
            elif hasattr(self, 'median_encoding_maps'):
                for num_col in self.numerical_features:
                    if num_col in df.columns:
                        key = f'{cat_col}_{num_col}_Median'
                        if key in self.median_encoding_maps:
                            df[key] = df[cat_col].map(
                                self.median_encoding_maps[key]
                            ).fillna(df[num_col].median() if num_col in df.columns else 0)
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders or use existing encoders
        
        Returns:
            Dataframe with encoded categorical features
        """
        df = df.copy()
        
        for col in self.categorical_features:
            if col not in df.columns:
                continue
            
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    unique_values = set(le.classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in unique_values else -1
                    )
        
        return df
    
    def scale_numerical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input dataframe
            fit: Whether to fit scaler or use existing scaler
        
        Returns:
            Dataframe with scaled numerical features
        """
        df = df.copy()
        
        if not self.numerical_features:
            return df
        
        # Get available numerical features (including newly created ones)
        available_num_features = [f for f in self.numerical_features if f in df.columns]
        
        # Also include newly created numerical features
        new_num_features = [col for col in df.columns 
                          if col not in self.categorical_features 
                          and col != self.target_column
                          and df[col].dtype in [np.int64, np.int32, np.float64, np.float32]
                          and col not in available_num_features
                          and not col.endswith('_Target_Mean')  # Don't scale target-encoded features
                          and not col.endswith('_Count')]  # Don't scale count-encoded features
        
        all_num_features = available_num_features + new_num_features
        
        if all_num_features:
            if fit:
                df[all_num_features] = self.scaler.fit_transform(df[all_num_features])
            else:
                df[all_num_features] = self.scaler.transform(df[all_num_features])
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dataframe with missing values filled
        """
        df = df.copy()
        
        # Numerical: fill with median
        for col in self.numerical_features:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Categorical: fill with mode
        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
        
        return df
    
    def transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None, fit: bool = True) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        Applies all transformations in the correct order.
        
        IMPORTANT: Target encoding is DISABLED by default to prevent data leakage.
        The y parameter is kept for API compatibility but target encoding won't be used
        unless explicitly enabled and out-of-fold encoding is properly implemented.
        
        Args:
            df: Input dataframe (features only, should NOT contain target column)
            y: Target series (optional, not used for target encoding by default)
            fit: Whether to fit transformers or use existing transformers
        
        Returns:
            Transformed dataframe (target column removed if present)
        """
        print("Starting feature engineering pipeline...")
        
        # Safety check: Remove target column if accidentally included (prevent leakage)
        if self.target_column in df.columns:
            print(f"Warning: Target column '{self.target_column}' found in features. Removing to prevent data leakage.")
            df = df.drop(columns=[self.target_column])
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Create interaction features
        df = self.create_interaction_features(df)
        
        # Step 3: Create statistical features (target encoding DISABLED by default)
        df = self.create_statistical_features(df, y=y, fit=fit)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical(df, fit=fit)
        
        # Step 5: Scale numerical features
        df = self.scale_numerical(df, fit=fit)
        
        # Step 6: Store feature order on fit (for prediction consistency)
        if fit:
            self.feature_order = df.columns.tolist()
            # Ensure target is not in feature order
            if self.target_column in self.feature_order:
                self.feature_order.remove(self.target_column)
        
        # Step 7: Ensure feature order matches training (for prediction)
        if not fit and self.feature_order is not None:
            # Add missing columns with zeros
            for col in self.feature_order:
                if col not in df.columns:
                    df[col] = 0
            # Reorder columns and remove extra columns
            df = df[self.feature_order]
        
        print(f"Feature engineering complete. Final shape: {df.shape}")
        if not self.use_target_encoding:
            print("Note: Target encoding is DISABLED to prevent data leakage. AUC should be realistic (0.7-0.9).")
        return df
    
    def save(self, path: Path):
        """
        Save feature engineer state to disk.
        
        Args:
            path: Path to save the feature engineer
        """
        save_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'target_column': self.target_column,
            'use_target_encoding': self.use_target_encoding,
            'target_encoding_maps': getattr(self, 'target_encoding_maps', {}),
            'count_encoding_maps': getattr(self, 'count_encoding_maps', {}),
            'median_encoding_maps': getattr(self, 'median_encoding_maps', {}),
            'feature_order': getattr(self, 'feature_order', None)
        }
        joblib.dump(save_data, path)
        print(f"Feature engineer saved to {path}")
    
    @classmethod
    def load(cls, path: Path):
        """
        Load feature engineer state from disk.
        
        Args:
            path: Path to load the feature engineer from
        
        Returns:
            Loaded FeatureEngineer instance
        """
        data = joblib.load(path)
        engineer = cls(
            categorical_features=data['categorical_features'],
            numerical_features=data['numerical_features'],
            target_column=data['target_column'],
            use_target_encoding=data.get('use_target_encoding', False)
        )
        engineer.label_encoders = data['label_encoders']
        engineer.scaler = data['scaler']
        engineer.target_encoding_maps = data.get('target_encoding_maps', {})
        engineer.count_encoding_maps = data.get('count_encoding_maps', {})
        engineer.median_encoding_maps = data.get('median_encoding_maps', {})
        engineer.feature_order = data.get('feature_order', None)
        print(f"Feature engineer loaded from {path}")
        return engineer
