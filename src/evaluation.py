"""
Evaluation Module
Handles model evaluation, metrics, and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import os


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate all evaluation metrics
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_prob (array): Prediction probabilities (optional)
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Add ROC AUC if probabilities are provided
    if y_prob is not None:
        if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
            # If probabilities for both classes, use spam class (index 1)
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # If single probability column
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def print_metrics(metrics, model_name='Model', dataset_name='Test'):
    """
    Print evaluation metrics in a formatted way
    
    Args:
        metrics (dict): Dictionary of metrics
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
    """
    print(f"\n{'='*60}")
    print(f"{model_name} - {dataset_name} Set Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} (spam class)")
    print(f"Recall:    {metrics['recall']:.4f} (spam class)")
    print(f"F1 Score:  {metrics['f1_score']:.4f} (spam class)")
    if 'roc_auc' in metrics:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"{'='*60}\n")


def plot_confusion_matrix(y_true, y_pred, labels=['Ham', 'Spam'], 
                         title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        labels (list): Class labels
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_prob, title='ROC Curve', save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_true (array): True labels
        y_prob (array): Prediction probabilities
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    # Handle probability array shape
    if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]  # Use spam class probabilities
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def compare_models(results_dict, metric='f1_score', save_path=None):
    """
    Compare multiple models using a bar chart
    
    Args:
        results_dict (dict): Dictionary of model results
        metric (str): Metric to compare
        save_path (str): Path to save the plot
    """
    model_names = []
    scores = []
    
    for model_name, metrics in results_dict.items():
        model_names.append(model_name.replace('_', ' ').title())
        scores.append(metrics[metric])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(model_names)), scores, color='steelblue', alpha=0.8)
    
    # Add value labels on top of bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_x() + bar.get_width()/2, score + 0.01, 
                f'{score:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}', 
             fontsize=14, fontweight='bold')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.ylim([0, 1.1])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison chart saved to {save_path}")
    
    plt.show()


def create_results_table(results_dict, save_path=None):
    """
    Create a comparison table of all models
    
    Args:
        results_dict (dict): Dictionary of model results
        save_path (str): Path to save the table
        
    Returns:
        pd.DataFrame: Results dataframe
    """
    data = []
    
    for model_name, metrics in results_dict.items():
        row = {
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1 Score': f"{metrics['f1_score']:.4f}",
        }
        if 'roc_auc' in metrics:
            row['ROC AUC'] = f"{metrics['roc_auc']:.4f}"
        data.append(row)
    
    df = pd.DataFrame(data)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Results table saved to {save_path}")
    
    return df


def analyze_errors(y_true, y_pred, texts, num_examples=5):
    """
    Analyze and display misclassified examples
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        texts (array): Text samples
        num_examples (int): Number of examples to display
    """
    # Find false positives (predicted spam, actually ham)
    fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]
    
    # Find false negatives (predicted ham, actually spam)
    fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]
    
    print(f"\n{'='*80}")
    print("ERROR ANALYSIS")
    print(f"{'='*80}")
    print(f"Total False Positives: {len(fp_indices)}")
    print(f"Total False Negatives: {len(fn_indices)}")
    print(f"{'='*80}\n")
    
    if len(fp_indices) > 0:
        print(f"FALSE POSITIVES (Predicted Spam, Actually Ham) - Top {min(num_examples, len(fp_indices))} examples:")
        print("-" * 80)
        for i, idx in enumerate(fp_indices[:num_examples], 1):
            print(f"\nExample {i}:")
            print(f"Text: {texts[idx][:200]}...")
            print()
    
    if len(fn_indices) > 0:
        print(f"\nFALSE NEGATIVES (Predicted Ham, Actually Spam) - Top {min(num_examples, len(fn_indices))} examples:")
        print("-" * 80)
        for i, idx in enumerate(fn_indices[:num_examples], 1):
            print(f"\nExample {i}:")
            print(f"Text: {texts[idx][:200]}...")
            print()


def generate_full_report(model_name, y_true, y_pred, y_prob=None, 
                        texts=None, output_dir='results'):
    """
    Generate a comprehensive evaluation report
    
    Args:
        model_name (str): Name of the model
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_prob (array): Prediction probabilities (optional)
        texts (array): Text samples (optional)
        output_dir (str): Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"GENERATING EVALUATION REPORT FOR {model_name.upper()}")
    print(f"{'='*80}\n")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, model_name, 'Test')
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Ham', 'Spam'], digits=4))
    
    # Confusion matrix
    cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, 
                         title=f'{model_name} - Confusion Matrix',
                         save_path=cm_path)
    
    # ROC curve
    if y_prob is not None:
        roc_path = os.path.join(output_dir, f'{model_name}_roc_curve.png')
        plot_roc_curve(y_true, y_prob,
                      title=f'{model_name} - ROC Curve',
                      save_path=roc_path)
    
    # Error analysis
    if texts is not None:
        analyze_errors(y_true, y_pred, texts)
    
    print(f"\n{'='*80}")
    print(f"REPORT GENERATION COMPLETE")
    print(f"{'='*80}\n")
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("This module is meant to be imported and used by other scripts.")
    print("See main.py for usage examples.")