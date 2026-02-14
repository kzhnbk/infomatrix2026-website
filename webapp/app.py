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

# Valid genes list (from training data)
VALID_GENES = [
    'ABL1', 'ACVR1', 'AGO2', 'AKT1', 'AKT2', 'AKT3', 'ALK', 'APC', 'AR', 'ARAF',
    'ARID1A', 'ARID1B', 'ARID2', 'ARID5B', 'ASXL1', 'ASXL2', 'ATM', 'ATR', 'ATRX',
    'AURKA', 'AURKB', 'AXIN1', 'AXL', 'B2M', 'BAP1', 'BARD1', 'BCL10', 'BCL2',
    'BCL2L11', 'BCOR', 'BRAF', 'BRCA1', 'BRCA2', 'BRD4', 'BRIP1', 'BTK', 'CARD11',
    'CARM1', 'CASP8', 'CBL', 'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CDH1', 'CDK12',
    'CDK4', 'CDK6', 'CDK8', 'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'CDKN2C',
    'CEBPA', 'CHEK2', 'CIC', 'CREBBP', 'CTCF', 'CTLA4', 'CTNNB1', 'DDR2', 'DICER1',
    'DNMT3A', 'DNMT3B', 'DUSP4', 'EGFR', 'EIF1AX', 'ELF3', 'EP300', 'EPAS1', 'EPCAM',
    'ERBB2', 'ERBB3', 'ERBB4', 'ERCC2', 'ERCC3', 'ERCC4', 'ERG', 'ERRFI1', 'ESR1',
    'ETV1', 'ETV6', 'EWSR1', 'EZH2', 'FAM58A', 'FANCA', 'FANCC', 'FAT1', 'FBXW7',
    'FGF19', 'FGF3', 'FGF4', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'FLT1', 'FLT3',
    'FOXA1', 'FOXL2', 'FOXO1', 'FOXP1', 'FUBP1', 'GATA3', 'GLI1', 'GNA11', 'GNAQ',
    'GNAS', 'H3F3A', 'HIST1H1C', 'HLA-A', 'HLA-B', 'HNF1A', 'HRAS', 'IDH1', 'IDH2',
    'IGF1R', 'IKBKE', 'IKZF1', 'IL7R', 'INPP4B', 'JAK1', 'JAK2', 'JUN', 'KDM5A',
    'KDM5C', 'KDM6A', 'KDR', 'KEAP1', 'KIT', 'KLF4', 'KMT2A', 'KMT2B', 'KMT2C',
    'KMT2D', 'KNSTRN', 'KRAS', 'LATS1', 'LATS2', 'MAP2K1', 'MAP2K2', 'MAP2K4',
    'MAP3K1', 'MAPK1', 'MDM2', 'MDM4', 'MED12', 'MEF2B', 'MEN1', 'MET', 'MGA',
    'MLH1', 'MPL', 'MSH2', 'MSH6', 'MTOR', 'MYC', 'MYCN', 'MYD88', 'MYOD1', 'NCOR1',
    'NF1', 'NF2', 'NFE2L2', 'NFKBIA', 'NKX2-1', 'NOTCH1', 'NOTCH2', 'NPM1', 'NRAS',
    'NSD1', 'NTRK1', 'NTRK2', 'NTRK3', 'NUP93', 'PAK1', 'PAX8', 'PBRM1', 'PDGFRA',
    'PDGFRB', 'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3R1', 'PIK3R2', 'PIK3R3', 'PIM1',
    'PMS1', 'PMS2', 'POLE', 'PPM1D', 'PPP2R1A', 'PPP6C', 'PRDM1', 'PTCH1', 'PTEN',
    'PTPN11', 'PTPRD', 'PTPRT', 'RAB35', 'RAC1', 'RAD21', 'RAD50', 'RAD51B',
    'RAD51C', 'RAD51D', 'RAD54L', 'RAF1', 'RARA', 'RASA1', 'RB1', 'RBM10', 'RET',
    'RHEB', 'RHOA', 'RICTOR', 'RIT1', 'RNF43', 'ROS1', 'RRAS2', 'RUNX1', 'RXRA',
    'RYBP', 'SDHB', 'SDHC', 'SETD2', 'SF3B1', 'SHOC2', 'SHQ1', 'SMAD2', 'SMAD3',
    'SMAD4', 'SMARCA4', 'SMARCB1', 'SMO', 'SOS1', 'SOX9', 'SPOP', 'SRC', 'SRSF2',
    'STAG2', 'STAT3', 'STK11', 'TCF3', 'TCF7L2', 'TERT', 'TET1', 'TET2', 'TGFBR1',
    'TGFBR2', 'TMPRSS2', 'TP53', 'TP53BP1', 'TSC1', 'TSC2', 'U2AF1', 'VEGFA', 'VHL',
    'WHSC1', 'WHSC1L1', 'XPO1', 'XRCC2', 'YAP1'
]

# Common variation types (most frequent from training data)
VALID_VARIATIONS = [
    'Amplification', 'Deletion', 'Truncating Mutations', 'Fusions', 'Overexpression',
    'Copy Number Loss', 'Wildtype', 'Promoter Mutations', 'Epigenetic Silencing',
    'Hypermethylation', 'Promoter Hypermethylation',
    # Common point mutations
    'V600E', 'L858R', 'T790M', 'G12D', 'G12V', 'G12C', 'G13D', 'Q61K', 'Q61R',
    'R132H', 'R172K', 'E545K', 'H1047R', 'D816V', 'D842V', 'F1174L', 'L1196M',
    'C797S', 'G719A', 'G719S', 'S768I', 'L861Q', 'T315I', 'E255K', 'M351T',
    # Exon mutations
    'Exon 19 deletion', 'Exon 20 insertion', 'Exon 11 mutations', 'Exon 9 mutations',
    'Exon 19 deletion/insertion', 'Exon 19 insertion', 'Exon 20 insertions',
    # Fusions
    'BCR-ABL1 Fusion', 'EML4-ALK Fusion', 'NPM-ALK Fusion', 'ROS1-CD74 Fusion',
    'BRAF-KIAA1549 Fusion', 'TMPRSS2-ERG Fusion', 'ETV6-NTRK3 Fusion',
    'FGFR3-TACC3 Fusion', 'KIF5B-RET Fusion', 'CCDC6-RET Fusion',
    # Splice variants
    'EGFRvIII', 'AR-V7', 'MET exon 14 skipping',
    # Common mutations by position
    'R175H', 'R248Q', 'R273H', 'R282W', 'Y220C', 'G245S', 'R249S', 'H179R',
    'K700E', 'R625H', 'N550K', 'S249C', 'Y373C', 'K650E', 'R248W', 'R273C'
]

# Minimum word count for description
MIN_DESCRIPTION_WORDS = 40

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


def count_words(text):
    """Count words in text"""
    if not text:
        return 0
    return len(text.split())


@app.route('/api/valid-inputs', methods=['GET'])
def get_valid_inputs():
    """API endpoint to get valid genes and variations for autocomplete"""
    return jsonify({
        'genes': VALID_GENES,
        'variations': VALID_VARIATIONS,
        'min_description_words': MIN_DESCRIPTION_WORDS
    })


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

        # Validate gene if provided
        if gene and gene.upper() not in [g.upper() for g in VALID_GENES]:
            error = f"Invalid gene '{gene}'. Please select a valid gene from the list."
        # Validate description word count
        elif not input_text:
            error = "Please provide text description for classification"
        elif count_words(input_text) < MIN_DESCRIPTION_WORDS:
            error = f"Description must contain at least {MIN_DESCRIPTION_WORDS} words. Current: {count_words(input_text)} words."
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
                         class_descriptions=CLASS_DESCRIPTIONS,
                         valid_genes=VALID_GENES,
                         valid_variations=VALID_VARIATIONS,
                         min_description_words=MIN_DESCRIPTION_WORDS)


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
