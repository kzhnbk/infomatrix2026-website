# AGI - Automatic Genetic Interpretation

A web-based machine learning system for classifying genetic variants in cancer research.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange.svg)](https://xgboost.readthedocs.io/)

## About

AGI (Automatic Genetic Interpretation) is a web application that uses machine learning to classify genetic variants associated with cancer. The system analyzes text descriptions of mutations and assigns them to one of nine clinical classification categories using an XGBoost model trained on variant data.

This project was developed as part of research into applying machine learning techniques to problems in medical genetics and oncology.

## Features

The application provides:

- **Variant Classification**: Classifies genetic variants into 9 categories including Likely Neutral, Uncertain Significance, Likely Pathogenic, Pathogenic, Likely Oncogenic, Oncogenic, Predicted Oncogenic, and Resistance to targeted therapy
- **Data Visualization**: Displays analysis of frequently mutated genes and their clinical significance
- **Model Comparison**: Benchmarks multiple machine learning algorithms on the classification task
- **Web Interface**: Interactive web-based interface for submitting variants and viewing results
- **REST API**: Programmatic access for integration with other tools

## Getting Started

### Requirements

- Python 3.8 or higher
- pip package manager

### Installation

Clone the repository:

```bash
git clone https://github.com/kzhnbk/infomatrix2026-website.git
cd infomatrix2026-website
```

Install the required dependencies:

```bash
cd webapp
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

The application will be available at `http://localhost:5000`.

## Usage

### Web Interface

To classify a variant using the web interface:

1. Open the application in your browser
2. Navigate to the Predict page
3. Enter the gene name (e.g., BRCA1)
4. Enter the specific variation (e.g., R1699Q)
5. Provide a text description of the mutation
6. Submit the form to receive classification results

The results will show the predicted class along with probability scores for all categories.

### API

The application exposes a REST API endpoint for programmatic access:

**POST** `/api/predict`

Request body:
```json
{
  "text": "Truncating mutation in BRCA1 gene associated with hereditary breast cancer"
}
```

Response:
```json
{
  "prediction": 5,
  "class_name": "Pathogenic",
  "description": "Variant is known to cause disease",
  "probabilities": {
    "1": 0.05,
    "2": 0.03,
    "3": 0.08,
    "4": 0.15,
    "5": 0.62,
    "6": 0.02,
    "7": 0.03,
    "8": 0.01,
    "9": 0.01
  }
}
```

## Model Performance

We evaluated several machine learning models on this classification task:

| Model | Log Loss | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|----------|---------|
| XGBoost | 1.156 | 75.0% | 0.718 | 0.878 |
| LightGBM | 1.429 | 73.5% | 0.701 | 0.881 |
| Random Forest | 1.280 | 72.0% | 0.682 | 0.851 |
| Logistic Regression | 2.066 | 65.2% | 0.571 | 0.671 |
| Naive Bayes | 32.285 | 53.0% | 0.471 | 0.563 |

XGBoost was selected as the final model based on its strong performance across multiple metrics.

## Dataset Analysis

The training data includes variants from genes frequently implicated in cancer. The most common genes in the dataset are:

- **BRCA1** (264 variants) - Tumor suppressor gene; mutations increase breast and ovarian cancer risk
- **TP53** (163 variants) - Commonly mutated in over 50% of cancers; linked to chemoresistance
- **EGFR** (141 variants) - Growth factor receptor; driver mutation in lung cancer
- **PTEN** (126 variants) - Regulator of cell growth pathways
- **BRCA2** (125 variants) - DNA repair gene; hereditary cancer susceptibility
- **KIT** (99 variants) - Receptor tyrosine kinase; target for imatinib therapy
- **BRAF** (93 variants) - Signaling protein; common in melanoma
- **ALK** (69 variants) - Fusion gene in lung cancer; target for crizotinib
- **ERBB2** (69 variants) - HER2 receptor; target for trastuzumab in breast cancer
- **PDGFRA** (60 variants) - Growth factor receptor; druggable target in GIST

## Technical Stack

- **Backend**: Flask web framework
- **Machine Learning**: XGBoost, scikit-learn
- **Text Processing**: TF-IDF vectorization, NLTK
- **Frontend**: HTML, CSS, JavaScript
- **Data Analysis**: NumPy, Pandas

## Project Structure

```
infomatrix/
├── webapp/
│   ├── app.py                      # Main Flask application
│   ├── requirements.txt            # Python dependencies
│   ├── models/
│   │   ├── train_model.py         # Training script
│   │   └── __init__.py
│   ├── saved_models/              # Serialized models
│   │   ├── xgboost_model.pkl
│   │   ├── tfidf_vectorizer.pkl
│   │   └── label_encoder.pkl
│   ├── static/                    # CSS and JavaScript
│   └── templates/                 # HTML templates
├── dataset/                       # Training data (not included)
├── mainnotebook.ipynb            # Analysis notebook
├── verify_inference.py           # Model testing script
└── cancer_mutation_analysis.pdf  # Project documentation
```

## License

This project was created for educational purposes as part of the Infomatrix 2026 competition.

## Contact

**Zhanibek Kassymkan**
- GitHub: [@kzhnbk](https://github.com/kzhnbk)
- Project Repository: [infomatrix2026-website](https://github.com/kzhnbk/infomatrix2026-website)
