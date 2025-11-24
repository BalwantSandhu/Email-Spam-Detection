"""
Test Script
Quick verification that all modules can be imported and basic functionality works
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("Testing Email Spam Detection Project Setup...\n")

# Test 1: Import all modules
print("Test 1: Importing modules...")
try:
    from src import data_preprocessing
    from src import feature_extraction
    from src import baseline_models
    from src import evaluation
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load dataset
print("\nTest 2: Loading dataset...")
try:
    df = data_preprocessing.load_dataset('data/email_spam.csv')
    print(f"✓ Dataset loaded: {len(df)} samples")
except Exception as e:
    print(f"✗ Failed to load dataset: {e}")
    sys.exit(1)

# Test 3: Preprocess dataset
print("\nTest 3: Preprocessing dataset...")
try:
    df = data_preprocessing.preprocess_dataset(df)
    print(f"✓ Dataset preprocessed")
    print(f"  - Spam: {df['label'].sum()} samples")
    print(f"  - Ham: {len(df) - df['label'].sum()} samples")
except Exception as e:
    print(f"✗ Preprocessing failed: {e}")
    sys.exit(1)

# Test 4: Check if splits exist
print("\nTest 4: Checking data splits...")
if os.path.exists('data/train.csv'):
    print("✓ Data splits already exist")
    train_df, dev_df, test_df = data_preprocessing.load_split_data('data')
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Dev: {len(dev_df)} samples")
    print(f"  - Test: {len(test_df)} samples")
else:
    print("ℹ Data splits not found. Run: python main.py --prepare-data")

# Test 5: Feature extraction test
print("\nTest 5: Testing feature extraction...")
try:
    extractor = feature_extraction.FeatureExtractor(method='tfidf', max_features=100)
    sample_texts = ["This is spam", "This is not spam", "Buy now!"]
    features = extractor.fit_transform(sample_texts)
    print(f"✓ Feature extraction works")
    print(f"  - Feature matrix shape: {features.shape}")
except Exception as e:
    print(f"✗ Feature extraction failed: {e}")
    sys.exit(1)

# Test 6: Baseline model initialization
print("\nTest 6: Testing baseline models...")
try:
    nb_model = baseline_models.BaselineModel('naive_bayes')
    svm_model = baseline_models.BaselineModel('svm')
    lr_model = baseline_models.BaselineModel('logistic_regression')
    print("✓ All baseline models initialized successfully")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED! ✓")
print("="*60)
print("\nProject setup is complete. You can now:")
print("1. Run: python main.py --prepare-data")
print("   (to split the dataset)")
print("2. Run: python main.py --baseline")
print("   (to train baseline models)")
print("3. Run: python main.py --full-pipeline")
print("   (to run everything including BERT, requires GPU)")
print("\nFor detailed analysis, open the Jupyter notebooks.")
print("="*60)