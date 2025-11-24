"""
Baseline Models Module
Implements Naive Bayes, SVM, and Logistic Regression classifiers
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import numpy as np


class BaselineModel:
    """
    Base class for baseline models
    """
    
    def __init__(self, model_type='naive_bayes', **kwargs):
        """
        Initialize baseline model
        
        Args:
            model_type (str): Type of model ('naive_bayes', 'svm', 'logistic_regression')
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        
    def _create_model(self, **kwargs):
        """
        Create the appropriate model
        
        Args:
            **kwargs: Model parameters
            
        Returns:
            sklearn model
        """
        if self.model_type == 'naive_bayes':
            alpha = kwargs.get('alpha', 1.0)
            return MultinomialNB(alpha=alpha)
        
        elif self.model_type == 'svm':
            C = kwargs.get('C', 1.0)
            max_iter = kwargs.get('max_iter', 1000)
            random_state = kwargs.get('random_state', 42)
            return LinearSVC(C=C, max_iter=max_iter, random_state=random_state, dual=False)
        
        elif self.model_type == 'logistic_regression':
            C = kwargs.get('C', 1.0)
            max_iter = kwargs.get('max_iter', 1000)
            random_state = kwargs.get('random_state', 42)
            solver = kwargs.get('solver', 'liblinear')
            return LogisticRegression(C=C, max_iter=max_iter, 
                                     random_state=random_state, solver=solver)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\nTraining {self.model_type.replace('_', ' ').title()}...")
        self.model.fit(X_train, y_train)
        print("Training complete!")
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            array: Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities (if available)
        
        Args:
            X: Features
            
        Returns:
            array: Prediction probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # For SVM, convert decision function to probabilities
            decision = self.model.decision_function(X)
            # Simple sigmoid transformation
            proba = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba, proba])
        else:
            raise NotImplementedError(f"{self.model_type} does not support probability predictions")
    
    def evaluate(self, X, y, dataset_name='Test'):
        """
        Evaluate the model
        
        Args:
            X: Features
            y: True labels
            dataset_name (str): Name of the dataset being evaluated
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\nEvaluating on {dataset_name} set...")
        
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Ham', 'Spam'], digits=4))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        print(f"\nTrue Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
        
        # Calculate metrics manually for spam class
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        return metrics
    
    def save(self, filepath):
        """
        Save the model
        
        Args:
            filepath (str): Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load the model
        
        Args:
            filepath (str): Path to load the model from
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return self


def train_naive_bayes(X_train, y_train, X_dev, y_dev, alpha=1.0):
    """
    Train and evaluate Multinomial Naive Bayes
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_dev: Development features
        y_dev: Development labels
        alpha (float): Smoothing parameter
        
    Returns:
        tuple: (model, dev_metrics)
    """
    model = BaselineModel('naive_bayes', alpha=alpha)
    model.train(X_train, y_train)
    
    # Evaluate on train set
    train_metrics = model.evaluate(X_train, y_train, 'Train')
    
    # Evaluate on dev set
    dev_metrics = model.evaluate(X_dev, y_dev, 'Dev')
    
    return model, dev_metrics


def train_svm(X_train, y_train, X_dev, y_dev, C=1.0, max_iter=1000):
    """
    Train and evaluate SVM
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_dev: Development features
        y_dev: Development labels
        C (float): Regularization parameter
        max_iter (int): Maximum iterations
        
    Returns:
        tuple: (model, dev_metrics)
    """
    model = BaselineModel('svm', C=C, max_iter=max_iter, random_state=42)
    model.train(X_train, y_train)
    
    # Evaluate on train set
    train_metrics = model.evaluate(X_train, y_train, 'Train')
    
    # Evaluate on dev set
    dev_metrics = model.evaluate(X_dev, y_dev, 'Dev')
    
    return model, dev_metrics


def train_logistic_regression(X_train, y_train, X_dev, y_dev, C=1.0, max_iter=1000):
    """
    Train and evaluate Logistic Regression
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_dev: Development features
        y_dev: Development labels
        C (float): Regularization parameter
        max_iter (int): Maximum iterations
        
    Returns:
        tuple: (model, dev_metrics)
    """
    model = BaselineModel('logistic_regression', C=C, max_iter=max_iter, 
                         random_state=42, solver='liblinear')
    model.train(X_train, y_train)
    
    # Evaluate on train set
    train_metrics = model.evaluate(X_train, y_train, 'Train')
    
    # Evaluate on dev set
    dev_metrics = model.evaluate(X_dev, y_dev, 'Dev')
    
    return model, dev_metrics


def compare_baseline_models(X_train, y_train, X_dev, y_dev, X_test, y_test):
    """
    Train and compare all baseline models
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_dev: Development features
        y_dev: Development labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary of models and their metrics
    """
    results = {}
    
    print("="*60)
    print("TRAINING BASELINE MODELS")
    print("="*60)
    
    # Naive Bayes
    print("\n1. MULTINOMIAL NAIVE BAYES")
    print("-"*60)
    nb_model, nb_dev_metrics = train_naive_bayes(X_train, y_train, X_dev, y_dev)
    nb_test_metrics = nb_model.evaluate(X_test, y_test, 'Test')
    results['naive_bayes'] = {
        'model': nb_model,
        'dev_metrics': nb_dev_metrics,
        'test_metrics': nb_test_metrics
    }
    
    # SVM
    print("\n2. SUPPORT VECTOR MACHINE (SVM)")
    print("-"*60)
    svm_model, svm_dev_metrics = train_svm(X_train, y_train, X_dev, y_dev)
    svm_test_metrics = svm_model.evaluate(X_test, y_test, 'Test')
    results['svm'] = {
        'model': svm_model,
        'dev_metrics': svm_dev_metrics,
        'test_metrics': svm_test_metrics
    }
    
    # Logistic Regression
    print("\n3. LOGISTIC REGRESSION")
    print("-"*60)
    lr_model, lr_dev_metrics = train_logistic_regression(X_train, y_train, X_dev, y_dev)
    lr_test_metrics = lr_model.evaluate(X_test, y_test, 'Test')
    results['logistic_regression'] = {
        'model': lr_model,
        'dev_metrics': lr_dev_metrics,
        'test_metrics': lr_test_metrics
    }
    
    # Summary
    print("\n" + "="*60)
    print("BASELINE MODELS COMPARISON (Test Set F1 Scores)")
    print("="*60)
    for model_name, result in results.items():
        f1 = result['test_metrics']['f1_score']
        print(f"{model_name.replace('_', ' ').title():30s}: {f1:.4f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_split_data, get_text_and_labels
    from feature_extraction import extract_tfidf_features
    
    print("Loading data...")
    train_df, dev_df, test_df = load_split_data()
    
    train_texts, train_labels = get_text_and_labels(train_df)
    dev_texts, dev_labels = get_text_and_labels(dev_df)
    test_texts, test_labels = get_text_and_labels(test_df)
    
    print("\nExtracting features...")
    train_X, dev_X, test_X, _ = extract_tfidf_features(
        train_texts, dev_texts, test_texts
    )
    
    print("\nTraining and comparing baseline models...")
    results = compare_baseline_models(
        train_X, train_labels,
        dev_X, dev_labels,
        test_X, test_labels
    )