"""Comprehensive reporting pipeline for model training and evaluation"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report, 
    confusion_matrix, roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
    
    Returns:
        Object with all numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


class ModelReportGenerator:
    """Generate comprehensive training and evaluation reports"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_data = {}
        
    def generate_training_report(self, 
                                trainer_results: Dict[str, Any],
                                feature_importance: Dict[str, Dict[str, float]],
                                cv_scores: Dict[str, np.ndarray],
                                test_results: Dict[str, Dict[str, float]],
                                training_time: float = None,
                                cv_metric_name: str = "cv_score",
                                best_model_metric: str = "roc_auc") -> Dict[str, Any]:
        """
        Generate comprehensive training report.
        
        Args:
            trainer_results: Results from model trainer
            feature_importance: Feature importance for each model
            cv_scores: Cross-validation scores
            test_results: Test set evaluation results
            training_time: Total training time in seconds
        
        Returns:
            Dictionary with report data
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'models_trained': list(test_results.keys()),
            'cv_metric': cv_metric_name,
            'best_model_metric': best_model_metric,
            'cross_validation': {},
            'test_performance': {},
            'best_model': None,
            'feature_importance': {}
        }
        
        # Cross-validation summary
        for model_name, scores in cv_scores.items():
            # Ensure all values are Python native types
            scores_list = scores.tolist() if isinstance(scores, np.ndarray) else list(scores)
            report['cross_validation'][model_name] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'scores': [convert_numpy_types(s) for s in scores_list]
            }
        
        # Test performance summary
        best_value = -1.0
        for model_name, metrics in test_results.items():
            # Convert numpy types to Python native types
            report['test_performance'][model_name] = {
                k: convert_numpy_types(v) for k, v in metrics.items()
            }
            metric_value = float(metrics.get(best_model_metric, metrics.get('roc_auc', 0.0)))
            if metric_value > best_value:
                best_value = metric_value
                report['best_model'] = model_name
        
        # Feature importance summary
        for model_name, importance_dict in feature_importance.items():
            if importance_dict:
                # Get top 20 features
                sorted_features = sorted(
                    importance_dict.items(), 
                    key=lambda x: float(x[1]),  # Ensure float conversion
                    reverse=True
                )[:20]
                # Convert numpy types in feature importance
                top_features_dict = {
                    k: convert_numpy_types(v) 
                    for k, v in sorted_features
                }
                report['feature_importance'][model_name] = {
                    'top_features': top_features_dict,
                    'total_features': len(importance_dict)
                }
        
        self.report_data = report
        return report
    
    def save_report_json(self, filename: str = 'training_report.json'):
        """
        Save report as JSON.
        Converts all numpy types to Python native types for JSON serialization.
        """
        report_path = self.output_dir / filename
        
        # Convert all numpy types to Python native types before saving
        serializable_data = convert_numpy_types(self.report_data)
        
        with open(report_path, 'w') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {report_path}")
    
    def generate_performance_plots(self, 
                                   y_true: np.ndarray,
                                   predictions: Dict[str, np.ndarray],
                                   probabilities: Dict[str, np.ndarray]):
        """
        Generate performance visualization plots.
        
        Args:
            y_true: True labels
            predictions: Dictionary of model predictions
            probabilities: Dictionary of model prediction probabilities
        """
        n_models = len(predictions)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 1. ROC Curves
        ax = axes[0]
        for model_name, proba in probabilities.items():
            fpr, tpr, _ = roc_curve(y_true, proba)
            auc_score = roc_auc_score(y_true, proba)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.4f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curves
        ax = axes[1]
        for model_name, proba in probabilities.items():
            precision, recall, _ = precision_recall_curve(y_true, proba)
            ax.plot(recall, precision, label=model_name, linewidth=2)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 3. Confusion Matrices (for best model)
        if predictions:
            best_model = max(probabilities.keys(), 
                           key=lambda x: roc_auc_score(y_true, probabilities[x]))
            ax = axes[2]
            cm = confusion_matrix(y_true, predictions[best_model])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Renewal', 'Renewal'],
                       yticklabels=['No Renewal', 'Renewal'])
            ax.set_title(f'Confusion Matrix - {best_model}', fontsize=14, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=12)
        
        # 4. Model Comparison Bar Chart
        ax = axes[3]
        metrics_to_plot = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
        model_names = list(predictions.keys())
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            if model_name in self.report_data.get('test_performance', {}):
                metrics = self.report_data['test_performance'][model_name]
                values = [metrics.get(m, 0) for m in metrics_to_plot]
                offset = (i - len(model_names) / 2 + 0.5) * width
                ax.bar(x + offset, values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_feature_importance_plots(self, 
                                          feature_importance: Dict[str, Dict[str, float]],
                                          top_n: int = 20):
        """
        Generate feature importance visualization.
        
        Args:
            feature_importance: Dictionary of model_name -> {feature: importance}
            top_n: Number of top features to display
        """
        n_models = len(feature_importance)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, importance_dict) in enumerate(feature_importance.items()):
            if not importance_dict:
                continue
                
            ax = axes[idx]
            
            # Sort features by importance
            sorted_features = sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_n]
            
            features = [f[0] for f in sorted_features]
            importances = [f[1] for f in sorted_features]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importances, align='center', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title(f'Top {top_n} Features - {model_name}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_cv_comparison_plot(self, cv_scores: Dict[str, np.ndarray]):
        """
        Generate cross-validation scores comparison.
        
        Args:
            cv_scores: Dictionary of model_name -> cv_scores_array
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data_to_plot = []
        labels = []
        for model_name, scores in cv_scores.items():
            data_to_plot.append(scores)
            labels.append(f"{model_name}\n(mean={np.mean(scores):.4f})")
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        cv_metric = self.report_data.get('cv_metric', 'CV Score')
        ax.set_ylabel(str(cv_metric), fontsize=12)
        ax.set_title('Cross-Validation Performance Comparison',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cv_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self) -> str:
        """
        Generate text summary report.
        
        Returns:
            Formatted string report
        """
        if not self.report_data:
            return "No report data available."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("INSURANCE RENEWAL PREDICTION - MODEL TRAINING REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nGenerated: {self.report_data.get('timestamp', 'N/A')}")
        
        if self.report_data.get('training_time_seconds'):
            minutes = self.report_data['training_time_seconds'] / 60
            report_lines.append(f"Training Time: {minutes:.2f} minutes ({self.report_data['training_time_seconds']:.2f} seconds)")
        
        # Cross-validation results
        report_lines.append("\n" + "-" * 80)
        report_lines.append("CROSS-VALIDATION RESULTS")
        report_lines.append("-" * 80)
        cv_metric = self.report_data.get('cv_metric', 'CV Score')
        for model_name, cv_data in self.report_data.get('cross_validation', {}).items():
            report_lines.append(f"\n{model_name.upper()}:")
            report_lines.append(f"  Mean {cv_metric}: {cv_data['mean']:.4f} (+/- {cv_data['std']*2:.4f})")
            report_lines.append(f"  Min: {cv_data['min']:.4f}, Max: {cv_data['max']:.4f}")
        
        # Test performance
        report_lines.append("\n" + "-" * 80)
        report_lines.append("TEST SET PERFORMANCE")
        report_lines.append("-" * 80)
        for model_name, metrics in self.report_data.get('test_performance', {}).items():
            report_lines.append(f"\n{model_name.upper()}:")
            for metric_name, value in metrics.items():
                report_lines.append(f"  {metric_name.capitalize()}: {value:.4f}")
        
        # Best model
        best_model = self.report_data.get('best_model')
        if best_model:
            report_lines.append("\n" + "-" * 80)
            report_lines.append(f"BEST MODEL: {best_model.upper()}")
            report_lines.append("-" * 80)
            best_metrics = self.report_data['test_performance'][best_model]
            for metric_name, value in best_metrics.items():
                report_lines.append(f"  {metric_name.capitalize()}: {value:.4f}")
        
        report_lines.append("\n" + "=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        report_path = self.output_dir / 'training_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"Summary report saved to {report_path}")
        return report_text
