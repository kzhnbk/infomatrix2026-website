"""
Quick script to train and save the XGBoost model
Run this before starting the Flask app
"""

import os
import sys

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from train_model import main

if __name__ == '__main__':
    print("Starting model training...")
    print("This will create a demonstration model for the web application.\n")

    try:
        model, vectorizer = main()
        print("\nModel training successful!")
        print("You can now run the Flask application with: python app.py")
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
