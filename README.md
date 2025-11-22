# Email Spam Detection Project

**Group Members:** Balwant Singh, Khushpreet Singh

## Project Overview

This project implements an email spam detection system using deep learning and traditional machine learning approaches. The goal is to classify emails as either SPAM or HAM (not spam) using various text classification techniques, with a focus on fine-tuning transformer models like BERT.

## Project Structure

```
email_spam_detection/
├── data/                          # Dataset directory
│   ├── email_spam.csv            # Original dataset
│   ├── train.csv                 # Training split
│   ├── dev.csv                   # Development/validation split
│   └── test.csv                  # Test split
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── feature_extraction.py    # TF-IDF and feature engineering
│   ├── baseline_models.py       # Naive Bayes, SVM, Logistic Regression
│   ├── bert_model.py            # BERT/DistilBERT implementation
│   └── evaluation.py            # Evaluation metrics and reporting
├── notebooks/                    # Jupyter notebooks
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_baseline_models.ipynb # Baseline model experiments
│   └── 03_bert_finetuning.ipynb # BERT fine-tuning experiments
├── models/                       # Saved model files
├── results/                      # Results, plots, and metrics
├── config/                       # Configuration files
│   └── config.yaml              # Hyperparameters and settings
├── requirements.txt             # Python dependencies
├── main.py                      # Main training script
└── README.md                    # This file
```

## Dataset

The dataset contains 84 email examples with the following columns:
- `title`: Email subject line
- `text`: Email body content
- `type`: Label (spam or not spam)

The dataset is split into:
- **Training set (70%)**: For model training
- **Development set (15%)**: For hyperparameter tuning and model selection
- **Test set (15%)**: For final evaluation

## Approach

### 1. Baseline Models (with TF-IDF features)
- **Multinomial Naive Bayes**: Classic baseline for text classification
- **Support Vector Machine (SVM)**: Strong baseline with linear kernel
- **Logistic Regression**: Additional baseline as suggested by professor

### 2. Deep Learning Model
- **DistilBERT**: Lightweight version of BERT for efficient fine-tuning
- Fine-tuned for sequence classification
- Captures bidirectional contextual relationships

## Evaluation Metrics

Primary metric: **F1 Score** (harmonic mean of Precision and Recall)

Additional metrics:
- Precision (for SPAM class)
- Recall (for SPAM class)
- Accuracy
- Confusion Matrix
- ROC-AUC Score

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing and Splitting
```bash
python -c "from src.data_preprocessing import split_dataset; split_dataset()"
```

### 2. Train Baseline Models
```bash
python main.py --model naive_bayes
python main.py --model svm
python main.py --model logistic_regression
```

### 3. Train BERT Model
```bash
python main.py --model bert --epochs 5 --batch_size 16
```

### 4. Evaluate All Models
```bash
python main.py --evaluate_all
```

### 5. Run Jupyter Notebooks
```bash
jupyter notebook
```

## Results

Results including precision, recall, F1-scores, and confusion matrices will be saved in the `results/` directory.

## Key Features

- ✅ Proper train/dev/test split (70/15/15)
- ✅ Multiple baseline models (Naive Bayes, SVM, Logistic Regression)
- ✅ TF-IDF feature extraction
- ✅ DistilBERT fine-tuning
- ✅ Comprehensive evaluation metrics
- ✅ Visualization of results
- ✅ Reproducible experiments with random seeds

## Future Improvements

- Data augmentation for better generalization
- Ensemble methods combining multiple models
- Cross-validation for more robust evaluation
- Hyperparameter optimization using grid/random search
- Analysis of misclassified examples

## References

- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
- Enron Email Dataset: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
- HuggingFace Transformers: https://huggingface.co/transformers/

## License

This project is for educational purposes as part of an NLP course.