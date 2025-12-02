"""
Email Spam Classifier Web Application
A simple Flask web UI to classify emails as spam or ham using trained models
"""

from flask import Flask, render_template, request, jsonify
import pickle
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import clean_text

app = Flask(__name__)

# Global variables for models
vectorizer = None
models = {}
model_names = ['SVM', 'Naive Bayes', 'Logistic Regression']

def load_models():
    """Load all trained models and vectorizer"""
    global vectorizer, models
    
    try:
        # Load TF-IDF vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✓ Vectorizer loaded")
        
        # Load SVM model (best model)
        with open('models/svm_model.pkl', 'rb') as f:
            models['SVM'] = pickle.load(f)
        print("✓ SVM model loaded")
        
        # Load Naive Bayes model
        with open('models/naive_bayes_model.pkl', 'rb') as f:
            models['Naive Bayes'] = pickle.load(f)
        print("✓ Naive Bayes model loaded")
        
        # Load Logistic Regression model
        with open('models/logistic_regression_model.pkl', 'rb') as f:
            models['Logistic Regression'] = pickle.load(f)
        print("✓ Logistic Regression model loaded")
        
        print("\n✓ All models loaded successfully!\n")
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Model files not found!")
        print(f"  Make sure you've run: python main.py --baseline")
        print(f"  Missing file: {e}\n")
        return False
    except Exception as e:
        print(f"\n✗ Error loading models: {e}\n")
        return False


def predict_email(email_text, model_name='SVM'):
    """
    Predict if an email is spam or ham
    
    Args:
        email_text (str): The email text to classify
        model_name (str): Which model to use ('SVM', 'Naive Bayes', 'Logistic Regression')
        
    Returns:
        dict: Prediction results with label, confidence, and model info
    """
    if not email_text or not email_text.strip():
        return {
            'error': 'Please enter some email text',
            'label': None,
            'confidence': None
        }
    
    try:
        # Clean the text
        cleaned_text = clean_text(email_text)
        
        # Convert to features
        features = vectorizer.transform([cleaned_text])
        
        # Get the selected model
        model = models.get(model_name, models['SVM'])
        
        # Make prediction
        prediction = int(model.predict(features)[0])  # Convert to Python int
        label = "SPAM" if prediction == 1 else "HAM"
        
        # Get probability/confidence if available
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0]
                confidence = float(proba[prediction] * 100)
            else:
                confidence = 85.0  # Default confidence
        except Exception as e:
            print(f"Warning: Could not calculate confidence: {e}")
            confidence = 85.0
        
        return {
            'label': label,
            'confidence': round(confidence, 2),
            'is_spam': bool(prediction == 1),  # Convert to Python bool
            'model_used': model_name,
            'cleaned_text_preview': cleaned_text[:100] + '...' if len(cleaned_text) > 100 else cleaned_text
        }
        
    except Exception as e:
        return {
            'error': f'Error during prediction: {str(e)}',
            'label': None,
            'confidence': None
        }


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', model_names=model_names)


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for email classification"""
    data = request.get_json()
    
    email_text = data.get('email_text', '')
    model_name = data.get('model_name', 'SVM')
    
    result = predict_email(email_text, model_name)
    return jsonify(result)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'vectorizer_loaded': vectorizer is not None
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("EMAIL SPAM CLASSIFIER - WEB APPLICATION")
    print("="*60)
    
    # Load models
    if load_models():
        print("Starting Flask server...")
        print("\n" + "="*60)
        print("🌐 Open your browser and go to:")
        print("   http://127.0.0.1:8080")
        print("   or")
        print("   http://localhost:8080")
        print("="*60 + "\n")
        print("Press Ctrl+C to stop the server\n")
        
        # Run the app
        app.run(debug=True, host='127.0.0.1', port=8080)
    else:
        print("\n⚠️  Cannot start server - models not loaded!")
        print("   Please train the models first by running:")
        print("   python main.py --baseline\n")
        sys.exit(1)