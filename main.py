"""
Main Training Script
Run all experiments and generate results
"""

import argparse
import os
import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import split_dataset, load_split_data, get_text_and_labels
from src.feature_extraction import extract_tfidf_features
from src.baseline_models import compare_baseline_models, BaselineModel
from src.evaluation import (
    generate_full_report, compare_models, create_results_table
)

# Try to import BERT, but it's optional
try:
    from src.bert_model import BERTSpamClassifier
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: PyTorch not installed. BERT functionality will be disabled.")


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_data_preparation():
    """Prepare and split the dataset"""
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION")
    print("="*80)
    
    if not os.path.exists('data/train.csv'):
        print("\nSplitting dataset into train/dev/test...")
        split_dataset(
            input_path='data/email_spam.csv',
            train_size=0.70,
            dev_size=0.15,
            test_size=0.15,
            random_state=42,
            output_dir='data'
        )
    else:
        print("\nDataset already split. Loading existing splits...")
        train_df, dev_df, test_df = load_split_data('data')
        print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    
    return True


def run_baseline_models(run_all=True):
    """Train and evaluate baseline models"""
    print("\n" + "="*80)
    print("STEP 2: BASELINE MODELS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_df, dev_df, test_df = load_split_data('data')
    train_texts, train_labels = get_text_and_labels(train_df)
    dev_texts, dev_labels = get_text_and_labels(dev_df)
    test_texts, test_labels = get_text_and_labels(test_df)
    
    # Extract TF-IDF features
    print("\nExtracting TF-IDF features...")
    train_X, dev_X, test_X, extractor = extract_tfidf_features(
        train_texts, dev_texts, test_texts,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Save feature extractor
    extractor.save('models/tfidf_vectorizer.pkl')
    
    # Train and compare baseline models
    print("\nTraining baseline models...")
    results = compare_baseline_models(
        train_X, train_labels,
        dev_X, dev_labels,
        test_X, test_labels
    )
    
    # Save models
    os.makedirs('models', exist_ok=True)
    for model_name, result in results.items():
        model_path = f'models/{model_name}_model.pkl'
        result['model'].save(model_path)
    
    # Generate reports
    os.makedirs('results', exist_ok=True)
    test_results = {}
    
    for model_name, result in results.items():
        print(f"\nGenerating report for {model_name}...")
        test_metrics = result['test_metrics']
        
        # Get predictions
        y_pred = test_metrics['predictions']
        
        # Get probabilities if available
        try:
            y_prob = result['model'].predict_proba(test_X)
        except:
            y_prob = None
        
        metrics = generate_full_report(
            model_name,
            test_labels,
            y_pred,
            y_prob,
            test_texts,
            output_dir='results'
        )
        
        test_results[model_name] = metrics
    
    # Create comparison visualizations
    print("\nCreating comparison visualizations...")
    compare_models(test_results, metric='f1_score', 
                  save_path='results/baseline_comparison_f1.png')
    compare_models(test_results, metric='accuracy',
                  save_path='results/baseline_comparison_accuracy.png')
    
    # Create results table
    results_df = create_results_table(test_results, 
                                     save_path='results/baseline_results.csv')
    
    return test_results


def run_bert_model(epochs=5, batch_size=16):
    """Train and evaluate BERT model"""
    if not BERT_AVAILABLE:
        print("\nERROR: PyTorch is not installed. Please install it first:")
        print("pip install torch transformers --break-system-packages")
        return None, None
    
    print("\n" + "="*80)
    print("STEP 3: BERT MODEL")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_df, dev_df, test_df = load_split_data('data')
    train_texts, train_labels = get_text_and_labels(train_df)
    dev_texts, dev_labels = get_text_and_labels(dev_df)
    test_texts, test_labels = get_text_and_labels(test_df)
    
    # Initialize BERT classifier
    print("\nInitializing DistilBERT classifier...")
    classifier = BERTSpamClassifier(
        model_name='distilbert-base-uncased',
        max_length=512,
        device='cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
    )
    
    # Train
    print("\nTraining BERT model...")
    history = classifier.train(
        train_texts, train_labels,
        dev_texts, dev_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=2e-5,
        warmup_steps=100
    )
    
    # Save model
    print("\nSaving BERT model...")
    classifier.save('models/bert_model')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loader = classifier.create_data_loader(
        test_texts, test_labels,
        batch_size=batch_size,
        shuffle=False
    )
    test_metrics = classifier.evaluate(test_loader)
    
    # Generate report
    print("\nGenerating BERT evaluation report...")
    y_pred = test_metrics['predictions']
    y_true = test_metrics['true_labels']
    
    metrics = generate_full_report(
        'bert_distilbert',
        y_true,
        y_pred,
        y_prob=None,  # BERT doesn't output probabilities easily in this setup
        texts=test_texts,
        output_dir='results'
    )
    
    return metrics, history


def run_full_pipeline(run_bert=False, bert_epochs=5, bert_batch_size=16):
    """Run the complete pipeline"""
    print("\n" + "#"*80)
    print("EMAIL SPAM DETECTION - FULL PIPELINE")
    print("#"*80)
    
    # Step 1: Data Preparation
    run_data_preparation()
    
    # Step 2: Baseline Models
    baseline_results = run_baseline_models()
    
    # Step 3: BERT Model (optional, requires GPU for reasonable training time)
    if run_bert:
        bert_results, bert_history = run_bert_model(
            epochs=bert_epochs,
            batch_size=bert_batch_size
        )
        
        # Compare all models
        all_results = baseline_results.copy()
        all_results['bert_distilbert'] = bert_results
        
        print("\nCreating final comparison...")
        compare_models(all_results, metric='f1_score',
                      save_path='results/final_comparison_f1.png')
        create_results_table(all_results,
                           save_path='results/final_results.csv')
    
    print("\n" + "#"*80)
    print("PIPELINE COMPLETE!")
    print("#"*80)
    print("\nResults saved in 'results/' directory")
    print("Models saved in 'models/' directory")
    print("\nNext steps:")
    print("1. Check results/ for evaluation metrics and visualizations")
    print("2. Run Jupyter notebooks for detailed analysis")
    print("3. Use saved models for predictions on new data")
    print("#"*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Email Spam Detection')
    parser.add_argument('--prepare-data', action='store_true',
                       help='Prepare and split the dataset')
    parser.add_argument('--baseline', action='store_true',
                       help='Train baseline models')
    parser.add_argument('--bert', action='store_true',
                       help='Train BERT model')
    parser.add_argument('--bert-epochs', type=int, default=5,
                       help='Number of epochs for BERT training')
    parser.add_argument('--bert-batch-size', type=int, default=16,
                       help='Batch size for BERT training')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run the complete pipeline')
    parser.add_argument('--evaluate-all', action='store_true',
                       help='Evaluate all saved models')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    if args.prepare_data:
        run_data_preparation()
    
    elif args.baseline:
        run_baseline_models()
    
    elif args.bert:
        run_bert_model(epochs=args.bert_epochs, batch_size=args.bert_batch_size)
    
    elif args.full_pipeline:
        run_full_pipeline(run_bert=True, 
                         bert_epochs=args.bert_epochs,
                         bert_batch_size=args.bert_batch_size)
    
    elif args.evaluate_all:
        print("Evaluating all saved models...")
        run_baseline_models()
    
    else:
        # Default: run baseline models only
        print("No specific option selected. Running baseline models...")
        print("Use --help to see all available options.")
        run_data_preparation()
        run_baseline_models()


if __name__ == "__main__":
    main()