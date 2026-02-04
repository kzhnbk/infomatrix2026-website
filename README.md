# 🧬 AGI - Automatic Genetic Interpretation

**AI-powered genetic variant classification system for cancer mutation analysis**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Project Overview

**AGI (Automatic Genetic Interpretation)** is a machine learning-powered web application for classifying genetic variants in the context of cancer. The system uses an **XGBoost** model to analyze textual descriptions of mutations and classifies them into 9 clinical categories.

### 🎯 Key Features

- 🔬 **Genetic Variant Classification** across 9 categories:
  - Likely Neutral
  - Likely Neutral (VUS)
  - Uncertain Significance
  - Likely Pathogenic
  - Pathogenic
  - Likely Oncogenic
  - Oncogenic
  - Predicted Oncogenic
  - Resistance (to targeted therapy)

- 📊 **Data Analysis**: Visualization of top genes and their roles in oncogenesis
- 🤖 **Model Comparison**: Performance evaluation of different ML algorithms
- 🌐 **Web Interface**: User-friendly Flask-based interface
- 🔌 **REST API**: Programmatic access to classification functions

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kzhnbk/infomatrix2026-website.git
   cd infomatrix2026-website
   ```

2. **Install dependencies**
   ```bash
   cd webapp
   pip install -r requirements.txt
   ```

3. **Run the web application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   ```
   http://localhost:5000
   ```

---

## 📁 Project Structure

```
infomatrix/
├── webapp/                     # Flask web application
│   ├── app.py                 # Main application file
│   ├── requirements.txt       # Python dependencies
│   ├── models/                # Machine learning models
│   │   ├── train_model.py    # Model training script
│   │   └── __init__.py
│   ├── saved_models/          # Trained models
│   │   ├── xgboost_model.pkl
│   │   ├── tfidf_vectorizer.pkl
│   │   └── label_encoder.pkl
│   ├── static/                # Static files (CSS, JS)
│   │   ├── css/
│   │   └── js/
│   └── templates/             # HTML templates
│       ├── index.html
│       ├── predict.html
│       ├── research.html
│       ├── models.html
│       ├── about.html
│       └── contact.html
├── dataset/                   # Dataset (not included in repository)
├── mainnotebook.ipynb         # Jupyter notebook with analysis
├── verify_inference.py        # Inference verification script
├── cancer_mutation_analysis.pdf  # Project documentation
└── README.md
```

---

## 🧪 Usage

### Web Interface

1. Navigate to the **Predict** page
2. Enter gene name (e.g., `BRCA1`)
3. Enter variation (e.g., `R1699Q`)
4. Enter text description of the mutation
5. Click **Classify**
6. Get classification results with probabilities for each class

### REST API

**Endpoint:** `POST /api/predict`

**Example Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Truncating mutation in BRCA1 gene associated with hereditary breast cancer"
  }'
```

**Example Response:**
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

---

## 🎓 Model Performance

| Model | Log Loss | Accuracy | F1 Macro | ROC AUC |
|--------|----------|----------|----------|---------|
| **XGBoost** | **1.156** | **75.0%** | **0.718** | **0.878** |
| LightGBM | 1.429 | 73.5% | 0.701 | 0.881 |
| Random Forest | 1.280 | 72.0% | 0.682 | 0.851 |
| Logistic Regression | 2.066 | 65.2% | 0.571 | 0.671 |
| Naive Bayes | 32.285 | 53.0% | 0.471 | 0.563 |

---

## 🔬 Top 10 Genes in Dataset

1. **BRCA1** (264) - Tumor suppressor; breast/ovarian cancer; PARP inhibitor target
2. **TP53** (163) - Mutated in >50% of cancers; associated with chemoresistance
3. **EGFR** (141) - Receptor tyrosine kinase; lung cancer driver; erlotinib/gefitinib target
4. **PTEN** (126) - Negative PI3K/AKT regulator; loss activates oncogenic pathways
5. **BRCA2** (125) - DNA repair; hereditary breast/ovarian cancer
6. **KIT** (99) - Receptor tyrosine kinase; GIST; imatinib-sensitive
7. **BRAF** (93) - Serine/threonine kinase; melanoma; vemurafenib target
8. **ALK** (69) - Receptor tyrosine kinase; lung cancer; crizotinib target
9. **ERBB2** (69) - HER2; breast cancer; trastuzumab target
10. **PDGFRA** (60) - Growth factor receptor; GIST; imatinib target

---

## 🛠️ Technologies

- **Backend:** Flask 2.3+
- **ML Framework:** XGBoost, scikit-learn
- **NLP:** NLTK, TF-IDF
- **Frontend:** HTML, CSS, JavaScript
- **Data Processing:** NumPy, Pandas

---

## 📝 License

This project was created for educational purposes as part of the Infomatrix 2026 competition.

---

## 👤 Author

**Zhanibek Kassymkan**

- GitHub: [@kzhnbk](https://github.com/kzhnbk)
- Project: [infomatrix2026-website](https://github.com/kzhnbk/infomatrix2026-website)

---

## 🙏 Acknowledgments

This project was developed as part of research on the application of machine learning in medical genetics and oncology.

---

**⭐ If you found this project helpful, please give it a star!**
