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

    IMPORTANT:
    - This project does NOT use target encoding in the main pipeline.
    - Target encoding can easily introduce leakage unless implemented as strict out-of-fold encoding.
    - We use leakage-safe alternatives (count/frequency + group aggregations on numeric features).
    """
    
    def __init__(self, categorical_features: List[str], 
                 numerical_features: List[str],
                 target_column: str = "Response"):
        """
        Initialize feature engineer.
        
        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            target_column: Name of target column
        """
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target_column = target_column
        
        # Encoders and transformers
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_order = None  # Store feature order for prediction consistency

        # Fitted artifacts for leakage-safe preprocessing
        # These are learned ONLY when fit=True and reused when fit=False.
        self.train_size_ = None
        self.numeric_fill_values_ = {}      # median per numerical feature
        self.categorical_fill_values_ = {}  # mode per categorical feature
        self.global_num_medians_ = {}       # fallback medians for unseen categories in aggregations
        self.global_num_means_ = {}         # fallback means for unseen categories in aggregations
        
        # Statistical encoding maps
        # Deprecated legacy field kept for backward compatibility when loading old artifacts.
        # Target encoding is not used in the current pipeline.
        self.target_encoding_maps = {}
        self.count_encoding_maps = {}
        self.median_encoding_maps = {}
        self.mean_encoding_maps = {}
        self.std_encoding_maps = {}

        # Date feature reference (optional)
        self.effective_date_ref_ = None
        
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

        # Additional log transforms for skewed numeric features (tree models benefit from monotonic transforms)
        if 'Total Claim Amount' in df.columns:
            df['Total_Claim_Amount_Log'] = np.log1p(df['Total Claim Amount'])
        if 'Monthly Premium Auto' in df.columns:
            df['Monthly_Premium_Auto_Log'] = np.log1p(df['Monthly Premium Auto'])
        
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

        # Optional date feature (if user keeps it in data)
        if 'Effective To Date' in df.columns:
            dt = pd.to_datetime(df['Effective To Date'], errors='coerce')
            df['Effective_To_Date_Month'] = dt.dt.month.fillna(0).astype(int)
            df['Effective_To_Date_Day'] = dt.dt.day.fillna(0).astype(int)
            df['Effective_To_Date_Weekday'] = dt.dt.weekday.fillna(0).astype(int)
            df['Effective_To_Date_Quarter'] = dt.dt.quarter.fillna(0).astype(int)
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame, y: Optional[pd.Series] = None, fit: bool = True) -> pd.DataFrame:
        """
        Create statistical aggregation features using count encoding and median encoding.
        
        NOTE:
        - Target encoding is intentionally NOT used here (leakage risk).
        - We generate leakage-safe statistical features only.
        
        Args:
            df: Input dataframe (features only, no target column)
            y: Target series (not used; target encoding is not part of this pipeline)
            fit: Whether to fit encoders or use existing mappings
        
        Returns:
            Dataframe with statistical features
        """
        df = df.copy()
        
        # Initialize encoding maps if fitting
        if fit and not hasattr(self, 'count_encoding_maps'):
            self.count_encoding_maps = {}
            self.median_encoding_maps = {}
            self.mean_encoding_maps = {}
            self.std_encoding_maps = {}

        # Store training-set reference statistics (used for fit=False to avoid test/val distribution peeking)
        if fit:
            self.train_size_ = int(len(df))
            self.global_num_medians_ = {
                col: float(df[col].median())
                for col in self.numerical_features
                if col in df.columns
            }
            self.global_num_means_ = {
                col: float(df[col].mean())
                for col in self.numerical_features
                if col in df.columns
            }

        denom = int(getattr(self, "train_size_", 0) or 0)
        if denom <= 0:
            # Backward compatibility: if an old artifact is loaded without train_size_,
            # fall back to current df size (not ideal but avoids crashing).
            denom = max(len(df), 1)
        
        # Create statistical features for categorical columns
        for cat_col in self.categorical_features:
            if cat_col not in df.columns:
                continue

            # Count encoding (SAFE - no target information used)
            if fit:
                count_map = df[cat_col].value_counts().to_dict()
                self.count_encoding_maps[cat_col] = count_map
                df[f'{cat_col}_Count'] = df[cat_col].map(count_map)
                # Frequency-style features (robust on low-cardinality categoricals)
                df[f'{cat_col}_Freq'] = df[f'{cat_col}_Count'] / denom
                df[f'{cat_col}_Count_Log'] = np.log1p(df[f'{cat_col}_Count'])
            elif hasattr(self, 'count_encoding_maps') and cat_col in self.count_encoding_maps:
                df[f'{cat_col}_Count'] = df[cat_col].map(
                    self.count_encoding_maps[cat_col]
                ).fillna(0)
                df[f'{cat_col}_Freq'] = df[f'{cat_col}_Count'] / denom
                df[f'{cat_col}_Count_Log'] = np.log1p(df[f'{cat_col}_Count'])
            
            # Aggregation encodings for numerical features grouped by category (SAFE - no target used)
            if fit:
                for num_col in self.numerical_features:
                    if num_col in df.columns:
                        grp = df.groupby(cat_col)[num_col]
                        median_map = grp.median().to_dict()
                        mean_map = grp.mean().to_dict()
                        std_map = grp.std().fillna(0).to_dict()

                        key_median = f'{cat_col}_{num_col}_Median'
                        key_mean = f'{cat_col}_{num_col}_Mean'
                        key_std = f'{cat_col}_{num_col}_Std'

                        self.median_encoding_maps[key_median] = median_map
                        self.mean_encoding_maps[key_mean] = mean_map
                        self.std_encoding_maps[key_std] = std_map

                        df[key_median] = df[cat_col].map(median_map)
                        df[key_mean] = df[cat_col].map(mean_map)
                        df[key_std] = df[cat_col].map(std_map)
            elif hasattr(self, 'median_encoding_maps'):
                for num_col in self.numerical_features:
                    if num_col in df.columns:
                        key_median = f'{cat_col}_{num_col}_Median'
                        key_mean = f'{cat_col}_{num_col}_Mean'
                        key_std = f'{cat_col}_{num_col}_Std'

                        if key_median in self.median_encoding_maps:
                            fallback_median = float(self.global_num_medians_.get(num_col, 0.0))
                            df[key_median] = df[cat_col].map(self.median_encoding_maps[key_median]).fillna(fallback_median)
                        if hasattr(self, 'mean_encoding_maps') and key_mean in self.mean_encoding_maps:
                            fallback_mean = float(self.global_num_means_.get(num_col, 0.0))
                            df[key_mean] = df[cat_col].map(self.mean_encoding_maps[key_mean]).fillna(fallback_mean)
                        if hasattr(self, 'std_encoding_maps') and key_std in self.std_encoding_maps:
                            df[key_std] = df[cat_col].map(self.std_encoding_maps[key_std]).fillna(0)
        
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
                          and not col.endswith('_Count')  # Don't scale count-encoded features
                          and not col.endswith('_Freq')   # Don't scale frequency features
                          and not col.endswith('_Count_Log')]  # Don't scale log-count features
        
        all_num_features = available_num_features + new_num_features
        
        if all_num_features:
            if fit:
                df[all_num_features] = self.scaler.fit_transform(df[all_num_features])
            else:
                df[all_num_features] = self.scaler.transform(df[all_num_features])
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dataframe with missing values filled
        """
        df = df.copy()

        # Learn fill statistics on training data only
        if fit:
            self.numeric_fill_values_ = {}
            self.categorical_fill_values_ = {}

            for col in self.numerical_features:
                if col in df.columns:
                    self.numeric_fill_values_[col] = float(df[col].median())

            for col in self.categorical_features:
                if col in df.columns:
                    mode_series = df[col].mode()
                    mode_val = mode_series.iloc[0] if len(mode_series) > 0 else 'Unknown'
                    self.categorical_fill_values_[col] = mode_val
        
        # Numerical: fill with median
        for col in self.numerical_features:
            if col in df.columns and df[col].isnull().any():
                fill_val = self.numeric_fill_values_.get(col, 0.0)
                df[col] = df[col].fillna(fill_val)
        
        # Categorical: fill with mode
        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().any():
                fill_val = self.categorical_fill_values_.get(col, 'Unknown')
                df[col] = df[col].fillna(fill_val)
        
        return df
    
    def transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None, fit: bool = True) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        Applies all transformations in the correct order.
        
        IMPORTANT:
        - Target encoding is not used in this pipeline (leakage risk).
        - The y parameter is kept for API compatibility and future safe extensions.
        
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
        df = self.handle_missing_values(df, fit=fit)
        
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
            # Legacy keys kept for backward compatibility (not used in current pipeline)
            'use_target_encoding': False,
            'target_encoding_maps': getattr(self, 'target_encoding_maps', {}),
            'count_encoding_maps': getattr(self, 'count_encoding_maps', {}),
            'median_encoding_maps': getattr(self, 'median_encoding_maps', {}),
            'mean_encoding_maps': getattr(self, 'mean_encoding_maps', {}),
            'std_encoding_maps': getattr(self, 'std_encoding_maps', {}),
            # New fitted artifacts (leakage-safe preprocessing)
            'train_size': getattr(self, 'train_size_', None),
            'numeric_fill_values': getattr(self, 'numeric_fill_values_', {}),
            'categorical_fill_values': getattr(self, 'categorical_fill_values_', {}),
            'global_num_medians': getattr(self, 'global_num_medians_', {}),
            'global_num_means': getattr(self, 'global_num_means_', {}),
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
            target_column=data['target_column']
        )
        engineer.label_encoders = data['label_encoders']
        engineer.scaler = data['scaler']
        # Legacy field (not used in current pipeline)
        engineer.target_encoding_maps = data.get('target_encoding_maps', {})
        engineer.count_encoding_maps = data.get('count_encoding_maps', {})
        engineer.median_encoding_maps = data.get('median_encoding_maps', {})
        engineer.mean_encoding_maps = data.get('mean_encoding_maps', {})
        engineer.std_encoding_maps = data.get('std_encoding_maps', {})
        engineer.feature_order = data.get('feature_order', None)
        # New fitted artifacts (backward-compatible defaults)
        engineer.train_size_ = data.get('train_size', None)
        engineer.numeric_fill_values_ = data.get('numeric_fill_values', {}) or {}
        engineer.categorical_fill_values_ = data.get('categorical_fill_values', {}) or {}
        engineer.global_num_medians_ = data.get('global_num_medians', {}) or {}
        engineer.global_num_means_ = data.get('global_num_means', {}) or {}
        print(f"Feature engineer loaded from {path}")
        return engineer
