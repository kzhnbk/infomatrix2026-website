"""
AGI - Automatic Genetic Interpretation - Flask Web Application
A machine learning web application for genetic variant classification
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import re
import string
import os

app = Flask(__name__)

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_models', 'xgboost_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'saved_models', 'tfidf_vectorizer.pkl')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'saved_models', 'label_encoder.pkl')

# Class descriptions for the 9 classification classes
# Colors: turquoise spectrum with red only for critical (Pathogenic)
CLASS_DESCRIPTIONS = {
    1: {
        'name': 'Likely Neutral',
        'description': 'Variant is likely to have no clinical significance',
        'color': '#14b8a6'
    },
    2: {
        'name': 'Likely Neutral (VUS)',
        'description': 'Variant of uncertain significance, leaning neutral',
        'color': '#0d9488'
    },
    3: {
        'name': 'Uncertain Significance',
        'description': 'Insufficient evidence to classify',
        'color': '#eab308'
    },
    4: {
        'name': 'Likely Pathogenic',
        'description': 'Variant is likely to be disease-causing',
        'color': '#f59e0b'
    },
    5: {
        'name': 'Pathogenic',
        'description': 'Variant is known to cause disease',
        'color': '#dc2626'
    },
    6: {
        'name': 'Likely Oncogenic',
        'description': 'Variant likely promotes cancer development',
        'color': '#0891b2'
    },
    7: {
        'name': 'Oncogenic',
        'description': 'Variant is a known cancer driver',
        'color': '#0e7490'
    },
    8: {
        'name': 'Predicted Oncogenic',
        'description': 'Computational prediction suggests oncogenic potential',
        'color': '#06b6d4'
    },
    9: {
        'name': 'Resistance',
        'description': 'Variant confers resistance to targeted therapy',
        'color': '#3d5a73'
    }
}

# Top genes data
TOP_GENES = [
    {'gene': 'BRCA1', 'frequency': 264, 'role': 'Tumor suppressor; breast/ovarian cancer; PARP inhibitor target'},
    {'gene': 'TP53', 'frequency': 163, 'role': 'Mutated in >50% of cancers; associated with chemoresistance'},
    {'gene': 'EGFR', 'frequency': 141, 'role': 'Receptor tyrosine kinase; lung cancer driver; erlotinib/gefitinib target'},
    {'gene': 'PTEN', 'frequency': 126, 'role': 'Negative PI3K/AKT regulator; loss activates oncogenic pathways'},
    {'gene': 'BRCA2', 'frequency': 125, 'role': 'DNA repair; hereditary breast/ovarian cancer'},
    {'gene': 'KIT', 'frequency': 99, 'role': 'Receptor tyrosine kinase; GIST; imatinib-sensitive'},
    {'gene': 'BRAF', 'frequency': 93, 'role': 'Serine/threonine kinase; melanoma; vemurafenib target'},
    {'gene': 'ALK', 'frequency': 69, 'role': 'Receptor tyrosine kinase; lung cancer; crizotinib target'},
    {'gene': 'ERBB2', 'frequency': 69, 'role': 'HER2; breast cancer; trastuzumab target'},
    {'gene': 'PDGFRA', 'frequency': 60, 'role': 'Growth factor receptor; GIST; imatinib target'}
]

# Model performance data
MODEL_PERFORMANCE = [
    {'model': 'XGBoost', 'log_loss': 1.156, 'accuracy': 0.750, 'f1_macro': 0.718, 'roc_auc': 0.878},
    {'model': 'LightGBM', 'log_loss': 1.429, 'accuracy': 0.735, 'f1_macro': 0.701, 'roc_auc': 0.881},
    {'model': 'Random Forest', 'log_loss': 1.280, 'accuracy': 0.720, 'f1_macro': 0.682, 'roc_auc': 0.851},
    {'model': 'Logistic Regression', 'log_loss': 2.066, 'accuracy': 0.652, 'f1_macro': 0.571, 'roc_auc': 0.671},
    {'model': 'Naive Bayes', 'log_loss': 32.285, 'accuracy': 0.530, 'f1_macro': 0.471, 'roc_auc': 0.563}
]


def clean_text(text):
    """
    Clean and preprocess text for model inference
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def load_model():
    """Load the trained model, vectorizer, and label encoder"""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return model, vectorizer, label_encoder
    except FileNotFoundError:
        return None, None, None


# Routes
@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')


@app.route('/about')
def about():
    """About the project page"""
    return render_template('about.html')


@app.route('/research')
def research():
    """Research and data analysis page"""
    return render_template('research.html', top_genes=TOP_GENES, class_descriptions=CLASS_DESCRIPTIONS)


@app.route('/models')
def models():
    """Model comparison page"""
    return render_template('models.html', model_performance=MODEL_PERFORMANCE)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction/inference page"""
    prediction = None
    probabilities = None
    error = None
    input_text = ""
    gene = ""
    variation = ""

    if request.method == 'POST':
        gene = request.form.get('gene', '').strip()
        variation = request.form.get('variation', '').strip()
        input_text = request.form.get('text', '').strip()

        if not input_text:
            error = "Please provide text description for classification"
        else:
            model, vectorizer, label_encoder = load_model()

            if model is None or vectorizer is None or label_encoder is None:
                error = "Model not loaded. Please ensure the model is trained and saved."
            else:
                try:
                    # Preprocess text
                    cleaned_text = clean_text(input_text)

                    # Vectorize
                    X = vectorizer.transform([cleaned_text])

                    # Predict
                    pred_encoded = model.predict(X)[0]
                    pred_proba = model.predict_proba(X)[0]

                    # Convert back to original class label
                    pred_class = label_encoder.inverse_transform([pred_encoded])[0]

                    # Get class info
                    prediction = {
                        'class': int(pred_class),
                        'info': CLASS_DESCRIPTIONS.get(int(pred_class), {
                            'name': f'Class {pred_class}',
                            'description': 'Unknown classification',
                            'color': '#6c757d'
                        })
                    }

                    # Get all probabilities
                    probabilities = []
                    original_classes = label_encoder.classes_
                    for i, prob in enumerate(pred_proba):
                        class_id = int(original_classes[i])
                        class_info = CLASS_DESCRIPTIONS.get(class_id, {'name': f'Class {class_id}', 'color': '#6c757d'})
                        probabilities.append({
                            'class': class_id,
                            'name': class_info['name'],
                            'probability': float(prob),
                            'color': class_info['color']
                        })

                    # Sort by probability
                    probabilities.sort(key=lambda x: x['probability'], reverse=True)

                except Exception as e:
                    error = f"Prediction error: {str(e)}"

    return render_template('predict.html',
                         prediction=prediction,
                         probabilities=probabilities,
                         error=error,
                         input_text=input_text,
                         gene=gene,
                         variation=variation,
                         class_descriptions=CLASS_DESCRIPTIONS)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    model, vectorizer, label_encoder = load_model()

    if model is None or vectorizer is None or label_encoder is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        text = data['text']
        cleaned_text = clean_text(text)
        X = vectorizer.transform([cleaned_text])

        pred_encoded = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]

        # Convert back to original class label
        pred_class = label_encoder.inverse_transform([pred_encoded])[0]

        original_classes = label_encoder.classes_
        probabilities = {int(original_classes[i]): float(prob) for i, prob in enumerate(pred_proba)}

        return jsonify({
            'prediction': int(pred_class),
            'class_name': CLASS_DESCRIPTIONS.get(int(pred_class), {}).get('name', f'Class {pred_class}'),
            'description': CLASS_DESCRIPTIONS.get(int(pred_class), {}).get('description', ''),
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
