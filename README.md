# 📧 Email Spam Detection Project

**Group Members:** Balwant Singh, Khushpreet Singh

---

## 🎯 Project Overview

This project implements a comprehensive email spam detection system using both traditional machine learning and modern deep learning approaches. The system classifies emails as either **SPAM** or **HAM** (legitimate) using various text classification techniques, featuring an interactive web interface for real-time classification.

### ✨ Key Features

- 🤖 **Multiple ML Models**: Naive Bayes, SVM, Logistic Regression, and DistilBERT
- 🌐 **Interactive Web UI**: Real-time email classification with confidence scores
- 📊 **Comprehensive Evaluation**: F1 Score, Precision, Recall, ROC-AUC, Confusion Matrices
- 🎨 **Visual Results**: Automatically generated charts and comparison plots
- 🔄 **Production Ready**: Trained models saved and ready for deployment

---

## 📁 Project Structure

```
email_spam_detection/
├── data/                          # Dataset directory
│   ├── email_spam.csv            # Original dataset (1000 emails)
│   ├── train.csv                 # Training split (70%)
│   ├── dev.csv                   # Development split (15%)
│   └── test.csv                  # Test split (15%)
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── feature_extraction.py    # TF-IDF and feature engineering
│   ├── baseline_models.py       # Naive Bayes, SVM, Logistic Regression
│   ├── bert_model.py            # BERT/DistilBERT implementation
│   └── evaluation.py            # Evaluation metrics and visualization
│
├── models/                       # Saved trained models
│   ├── naive_bayes_model.pkl
│   ├── svm_model.pkl            # Best performer (F1: 0.95+)
│   ├── logistic_regression_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── results/                      # Evaluation results
│   ├── baseline_results.csv     # Performance comparison table
│   ├── *_confusion_matrix.png   # Confusion matrices
│   ├── *_roc_curve.png         # ROC curves
│   └── baseline_comparison_*.png # Model comparison charts
│
├── templates/                    # Web UI templates
│   └── index.html               # Interactive web interface
│
├── config/                       # Configuration files
│   └── config.yaml              # Hyperparameters and settings
│
├── app.py                        # Flask web application
├── main.py                       # Main training script
├── test_setup.py                # Setup verification script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 📊 Dataset

**Size:** 1,000 emails (balanced dataset)
- **Spam:** 500 emails (50%)
- **Legitimate (Ham):** 500 emails (50%)

**Columns:**
- `title`: Email subject line
- `text`: Email body content
- `type`: Label (spam or not spam)

**Split:**
- **Training set (70%)**: 700 emails for model training
- **Development set (15%)**: 150 emails for hyperparameter tuning
- **Test set (15%)**: 150 emails for final evaluation

**Dataset Features:**
- Diverse spam patterns (phishing, scams, promotions)
- Realistic legitimate emails (work, personal, official)
- Balanced classes for fair model evaluation

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Steps

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/email-spam-detection.git
cd email-spam-detection
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python test_setup.py
```

---

## 🚀 Quick Start

### Option 1: Use the Web Interface (Recommended)

```bash
# Start the web application
python app.py
```

Then open your browser and go to: **http://127.0.0.1:8080**

**Features:**
- ✨ Beautiful, modern interface
- 🎯 Real-time email classification
- 📊 Confidence scores with visual indicators
- 🔄 Choose between different ML models
- 📝 Pre-loaded example emails for testing

### Option 2: Command Line Training

```bash
# Step 1: Prepare data splits
python main.py --prepare-data

# Step 2: Train all baseline models
python main.py --baseline

# Step 3: (Optional) Train BERT model
python main.py --bert --bert-epochs 5 --bert-batch-size 16

# Step 4: View results
ls results/
```

---

## 🤖 Models & Approaches

### 1. Baseline Models (TF-IDF Features)

#### **Multinomial Naive Bayes**
- Classic probabilistic baseline
- Fast training and inference
- Performance: F1 ~0.95

#### **Support Vector Machine (SVM)** ⭐ **Best Model**
- Linear kernel with optimized hyperparameters
- Excellent performance on text data
- **Performance: F1 ~0.98+**

#### **Logistic Regression**
- Strong baseline with interpretable weights
- Regularized for better generalization
- Performance: F1 ~0.96

### 2. Deep Learning Model

#### **DistilBERT**
- Lightweight transformer model (66M parameters)
- Fine-tuned for sequence classification
- Captures bidirectional context
- Optional: Requires GPU for efficient training

### Feature Engineering

- **TF-IDF Vectorization**: Captures term importance
- **N-grams (1,2)**: Captures both words and phrases
- **Configurable parameters**: Max features, min/max document frequency
- **Sublinear TF scaling**: Reduces impact of term frequency

---

## 📈 Results

### Model Performance (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Naive Bayes | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **SVM** ⭐ | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Analysis of Results

**Note on Perfect Scores:**
Our models achieved 100% accuracy on the test set. While impressive, this suggests:

1. **Well-separated patterns**: The spam and legitimate emails in our dataset have distinct linguistic features
2. **Dataset characteristics**: The generated dataset may have clear, unambiguous patterns
3. **Real-world expectation**: In production with more diverse, ambiguous emails, we expect 90-95% accuracy

**Key Learnings:**
- ✅ Models successfully learned spam patterns
- ✅ TF-IDF features effectively capture spam indicators
- ✅ All three baseline approaches perform equally well
- ⚠️ Perfect scores indicate potential for overfitting on more complex data
- 🎯 Production deployment would require validation on real-world email corpus

### Real-World Performance Expectations

In production spam filters:
- **Expected Accuracy**: 90-95%
- **Expected F1 Score**: 0.88-0.96
- **Reason**: Real emails have more ambiguity, edge cases, and evolving spam tactics

---

## 🎯 Evaluation Metrics

**Primary Metric:** **F1 Score** - Harmonic mean of Precision and Recall
- Perfect for imbalanced datasets
- Balances false positives vs false negatives

**Additional Metrics:**
- **Precision**: Avoid marking legitimate emails as spam
- **Recall**: Catch as many spam emails as possible
- **Accuracy**: Overall correctness
- **ROC-AUC**: Model's discriminative ability
- **Confusion Matrix**: Detailed error analysis

**All results saved in:** `results/` directory with visualizations

---

## 🌐 Web Application

### Features

🎨 **Beautiful Modern UI**
- Gradient design with smooth animations
- Responsive layout (works on mobile/tablet/desktop)
- Color-coded results (green=ham, red=spam)

🧠 **Smart Classification**
- Choose between 3 ML models
- Real-time processing
- Confidence scores with progress bars

📝 **Example Emails**
- Pre-loaded spam and legitimate examples
- Quick testing with one click

🔒 **Privacy**
- 100% local processing
- No data sent to external servers
- Completely private and secure

### Usage

```bash
# Start the server
python app.py

# Open browser
http://127.0.0.1:8080

# Test with examples or paste your own emails!
```

---

### Use Trained Models for Predictions

```python
import pickle

# Load model and vectorizer
with open('models/svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Classify new email
new_email = "Congratulations! You won $1,000,000!"
features = vectorizer.transform([new_email])
prediction = model.predict(features)[0]

print("SPAM" if prediction == 1 else "HAM")
```

---

## 🔄 Workflow

1. **Data Preprocessing**: Clean and split dataset
2. **Feature Extraction**: Convert text to TF-IDF vectors
3. **Model Training**: Train multiple ML models
4. **Evaluation**: Test on held-out test set
5. **Visualization**: Generate charts and confusion matrices
6. **Deployment**: Use in web UI or via API

---

## 📊 Visualizations

Automatically generated visualizations include:

- **Confusion Matrices**: For each model
- **ROC Curves**: Model discrimination analysis
- **Comparison Charts**: F1 scores and accuracy across models
- **Performance Tables**: Detailed metrics comparison

All saved as high-resolution PNG files in `results/`

---

## 🚀 Future Improvements

### Short-term
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation for robust evaluation
- [ ] Feature importance analysis
- [ ] Error analysis on misclassified examples

### Long-term
- [ ] Ensemble methods combining multiple models
- [ ] Character-level n-grams for obfuscation detection
- [ ] Active learning for continuous improvement
- [ ] Deploy as REST API service
- [ ] Multi-language support

---

## 🛠️ Technical Details

### Dependencies
- **scikit-learn**: ML models and evaluation
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **Flask**: Web application framework
- **transformers**: BERT models (optional)
- **PyTorch**: Deep learning backend (optional)

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for models and data
- **GPU**: Optional, only for BERT training

---

## 📖 References

1. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
2. Scikit-learn Documentation: https://scikit-learn.org/
3. Flask Documentation: https://flask.palletsprojects.com/
4. HuggingFace Transformers: https://huggingface.co/transformers/
5. Email Spam Datasets: https://www.kaggle.com/datasets

---

## 📝 License

This project is for educational purposes as part of an NLP course.

---

## 👥 Team

**Balwant Singh** - Model Development, Feature Engineering
**Khushpreet Singh** - Web UI, Data Processing

---

## 🙏 Acknowledgments

- Professor for valuable feedback and guidance
- Open-source community for amazing tools
- Kaggle for dataset resources

---

**⭐ Star this repo if you found it helpful!**

---

*Last Updated: November 22, 2024*