"""
XGBoost Model Training Script for AGI - Automatic Genetic Interpretation

This script trains an XGBoost model on the MSK Cancer Treatment dataset
and saves the model, vectorizer, and label encoder for use in the web application.
"""

import os
import re
import string
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize

import xgboost as xgb

warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(WEBAPP_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, 'dataset')
SAVED_MODELS_DIR = os.path.join(WEBAPP_DIR, 'saved_models')

# Ensure saved_models directory exists
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


def clean_text(text):
    """
    Comprehensive text cleaning function
    
    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove special characters
    4. Remove extra whitespace
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


def load_data():
    """Load and merge the training datasets"""
    print("Loading datasets...")
    
    # Load variants data
    variants_path = os.path.join(DATASET_DIR, 'training_variants')
    if os.path.exists(variants_path + '.csv'):
        variants_path += '.csv'
    
    # Load text data
    text_path = os.path.join(DATASET_DIR, 'training_text')
    if os.path.exists(text_path + '.csv'):
        text_path += '.csv'
    
    # Read variants
    print(f"  Reading variants from {variants_path}...")
    try:
        # Check if file is comma, tab or other separated
        with open(variants_path, 'r') as f:
            first_line = f.readline()
        
        sep = ','
        if '\t' in first_line and ',' not in first_line:
            sep = '\t'
            
        df_variants = pd.read_csv(variants_path, sep=sep)
        print(f"  Variants loaded. Shape: {df_variants.shape}")
    except Exception as e:
        print(f"  Error loading variants: {e}")
        # Try finding the file in dataset dir if path was wrong
        variants_path = os.path.join(DATASET_DIR, 'training_variants')
        try:
            df_variants = pd.read_csv(variants_path)
            print(f"  Variants loaded from fallback. Shape: {df_variants.shape}")
        except Exception as e2:
            print(f"  Critical error loading variants: {e2}")
            raise e2

    # Read text data
    print(f"  Reading text data from {text_path}...")
    try:
        # The text file uses || as separator and has ID||Text format
        # It's better to explicitly use the correct path without extension if original failed
        real_text_path = text_path
        if not os.path.exists(text_path) and os.path.exists(os.path.join(DATASET_DIR, 'training_text')):
            real_text_path = os.path.join(DATASET_DIR, 'training_text')
        
        # Determine skip rows
        skiprows = 1
        with open(real_text_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            if 'ID' not in header:
                skiprows = 0
        
        df_text = pd.read_csv(real_text_path, sep='\|\|', engine='python', 
                              header=None, names=['ID', 'Text'], skiprows=skiprows)
        print(f"  Text data loaded. Shape: {df_text.shape}")
        
    except Exception as e:
        print(f"  Error loading text data: {e}")
        raise e

    # Merge datasets on ID
    print("  Merging datasets...")
    try:
        # Ensure ID is int in both
        df_variants['ID'] = df_variants['ID'].astype(int)
        df_text['ID'] = df_text['ID'].astype(int)
        
        df = pd.merge(df_variants, df_text, on='ID', how='inner')
    except Exception as e_merge:
        print(f"  Merge failed: {e_merge}")
        print(f"  Variants columns: {df_variants.columns}")
        print(f"  Text columns: {df_text.columns}")
        print(f"  Variants ID dtypes: {df_variants['ID'].dtype}")
        print(f"  Text ID dtypes: {df_text['ID'].dtype}")
        raise e_merge
    
    print(f"  Merged dataset shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    return df


def preprocess_data(df):
    """Preprocess the data - clean text and prepare features"""
    print("\nPreprocessing data...")
    
    # Handle missing text values
    df['Text'] = df['Text'].fillna('')
    
    # Clean text
    print("  Cleaning text...")
    df['cleaned_text'] = df['Text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 10]
    
    print(f"  Dataset after cleaning: {df.shape}")
    
    return df


def create_features(df, max_features=5000):
    """Create TF-IDF features from text data"""
    print("\nCreating TF-IDF features...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X = vectorizer.fit_transform(df['cleaned_text'])
    
    print(f"  TF-IDF Features Shape: {X.shape}")
    print(f"  Vocabulary Size: {len(vectorizer.vocabulary_)}")
    
    return X, vectorizer


def train_xgboost(X_train, y_train, class_weights=None):
    """Train XGBoost classifier"""
    print("\nTraining XGBoost model...")
    
    # Convert class weights to sample weights
    if class_weights is not None:
        sample_weights = np.array([class_weights.get(y, 1.0) for y in y_train])
    else:
        sample_weights = None
    
    # Determine number of classes
    n_classes = len(np.unique(y_train))
    
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        tree_method='hist',
        n_jobs=-1,
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights)
    print("  XGBoost training complete!")
    
    return model


def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate the trained model"""
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    logloss = log_loss(y_test, y_pred_proba)
    
    # ROC AUC
    try:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        if y_test_bin.shape[1] > 1:
            roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')
        else:
            roc_auc = 0.0
    except:
        roc_auc = 0.0
    
    print(f"\n  === Model Performance ===")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  F1-Score (Macro):  {f1_macro:.4f}")
    print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"  Log Loss:          {logloss:.4f}")
    print(f"  ROC AUC:           {roc_auc:.4f}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'log_loss': logloss,
        'roc_auc': roc_auc
    }


def save_model(model, vectorizer, label_encoder):
    """Save model, vectorizer, and label encoder"""
    print("\nSaving model artifacts...")
    
    model_path = os.path.join(SAVED_MODELS_DIR, 'xgboost_model.pkl')
    vectorizer_path = os.path.join(SAVED_MODELS_DIR, 'tfidf_vectorizer.pkl')
    encoder_path = os.path.join(SAVED_MODELS_DIR, 'label_encoder.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"  Model saved to: {model_path}")
    print(f"  Vectorizer saved to: {vectorizer_path}")
    print(f"  Label encoder saved to: {encoder_path}")
    
    return model_path, vectorizer_path, encoder_path


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("AGI - XGBoost Model Training Pipeline")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Preprocess
    df = preprocess_data(df)
    
    # Create features
    X, vectorizer = create_features(df)
    
    # Prepare target variable
    print("\nPreparing target variable...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Class'])
    
    print(f"  Classes: {label_encoder.classes_}")
    print(f"  Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        original_class = label_encoder.inverse_transform([u])[0]
        print(f"    Class {original_class}: {c} samples")
    
    # Calculate class weights
    print("\nCalculating class weights...")
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    
    # Train-test split
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")
    
    # Train model
    model = train_xgboost(X_train, y_train, class_weights)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, label_encoder)
    
    # Save
    save_model(model, vectorizer, label_encoder)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return model, vectorizer


if __name__ == '__main__':
    main()
