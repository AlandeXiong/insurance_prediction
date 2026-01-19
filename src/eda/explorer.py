"""Comprehensive EDA analysis"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class DataExplorer:
    """Comprehensive EDA for insurance renewal prediction"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load dataset"""
        df = pd.read_csv(data_path)
        print(f"Dataset shape: {df.shape}")
        return df
    
    def basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        print("\n" + "="*50)
        print("BASIC DATASET INFORMATION")
        print("="*50)
        print(f"Shape: {info['shape']}")
        print(f"Memory Usage: {info['memory_usage_mb']:.2f} MB")
        print(f"Duplicates: {info['duplicates']}")
        print("\nMissing Values:")
        missing = {k: v for k, v in info['missing_values'].items() if v > 0}
        if missing:
            for col, count in missing.items():
                print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            print("  No missing values!")
        
        return info
    
    def target_analysis(self, df: pd.DataFrame, target_col: str):
        """Analyze target variable"""
        print("\n" + "="*50)
        print("TARGET VARIABLE ANALYSIS")
        print("="*50)
        
        target_counts = df[target_col].value_counts()
        target_props = df[target_col].value_counts(normalize=True)
        
        print(f"\nTarget Distribution:")
        for val, count in target_counts.items():
            prop = target_props[val]
            print(f"  {val}: {count} ({prop*100:.2f}%)")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        sns.countplot(data=df, x=target_col, ax=axes[0], palette='viridis')
        axes[0].set_title('Target Variable Distribution (Count)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel(target_col, fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        
        # Proportion plot
        target_props.plot(kind='bar', ax=axes[1], color=['#1f77b4', '#ff7f0e'])
        axes[1].set_title('Target Variable Distribution (Proportion)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel(target_col, fontsize=12)
        axes[1].set_ylabel('Proportion', fontsize=12)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return target_counts, target_props
    
    def numerical_analysis(self, df: pd.DataFrame, numerical_cols: list):
        """Analyze numerical features"""
        print("\n" + "="*50)
        print("NUMERICAL FEATURES ANALYSIS")
        print("="*50)
        
        if not numerical_cols:
            print("No numerical features to analyze")
            return
        
        # Statistics
        stats = df[numerical_cols].describe()
        print("\nDescriptive Statistics:")
        print(stats)
        
        # Distribution plots
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numerical_cols):
            ax = axes[idx]
            df[col].hist(bins=50, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation matrix
        corr_matrix = df[numerical_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Numerical Features Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return stats
    
    def categorical_analysis(self, df: pd.DataFrame, categorical_cols: list, target_col: str):
        """Analyze categorical features"""
        print("\n" + "="*50)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("="*50)
        
        if not categorical_cols:
            print("No categorical features to analyze")
            return
        
        # Value counts for each categorical feature
        for col in categorical_cols:
            if col == target_col:
                continue
            print(f"\n{col}:")
            print(df[col].value_counts())
            print(f"Unique values: {df[col].nunique()}")
        
        # Target distribution by categorical features
        n_cols = min(3, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(categorical_cols):
            if col == target_col:
                continue
            ax = axes[idx]
            
            # Create crosstab
            crosstab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
            
            # Plot
            crosstab.plot(kind='bar', ax=ax, stacked=False, color=['#1f77b4', '#ff7f0e'])
            ax.set_title(f'Target Distribution by {col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Percentage', fontsize=10)
            ax.legend(title=target_col, fontsize=9)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide extra subplots
        visible_count = len([c for c in categorical_cols if c != target_col])
        for idx in range(visible_count, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'categorical_target_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def feature_target_relationship(self, df: pd.DataFrame, numerical_cols: list, 
                                   categorical_cols: list, target_col: str):
        """Analyze relationships between features and target"""
        print("\n" + "="*50)
        print("FEATURE-TARGET RELATIONSHIPS")
        print("="*50)
        
        # Numerical features vs target
        if numerical_cols:
            n_cols = min(3, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            for idx, col in enumerate(numerical_cols):
                ax = axes[idx]
                
                # Box plot
                df.boxplot(column=col, by=target_col, ax=ax, grid=True)
                ax.set_title(f'{col} by {target_col}', fontsize=12, fontweight='bold')
                ax.set_xlabel(target_col, fontsize=10)
                ax.set_ylabel(col, fontsize=10)
                ax.get_figure().suptitle('')  # Remove default title
            
            # Hide extra subplots
            for idx in range(len(numerical_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'numerical_target_relationship.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self, df: pd.DataFrame, target_col: str, 
                       numerical_cols: list, categorical_cols: list):
        """Generate comprehensive EDA report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE EDA REPORT GENERATION")
        print("="*70)
        
        # Basic info
        self.basic_info(df)
        
        # Target analysis
        self.target_analysis(df, target_col)
        
        # Numerical analysis
        self.numerical_analysis(df, numerical_cols)
        
        # Categorical analysis
        self.categorical_analysis(df, categorical_cols, target_col)
        
        # Feature-target relationships
        self.feature_target_relationship(df, numerical_cols, categorical_cols, target_col)
        
        print("\n" + "="*70)
        print(f"EDA Report saved to: {self.output_dir}")
        print("="*70)
