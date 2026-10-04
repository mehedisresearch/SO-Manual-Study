import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from scipy.sparse import hstack
import joblib

def run_tfidf_baseline():
    print("\n--- Running TF-IDF Statistical Baseline Analysis ---")
    
    # 1. Load Data
    train_df = pd.read_parquet('data/processed/train_bert.parquet')
    test_df = pd.read_parquet('data/processed/test.parquet')

    y_train = (train_df['Label'] == 'Rejected').astype(int)
    y_test = (test_df['Label'] == 'Rejected').astype(int)

    # 2. Vectorization Setup
    # We use a mix of Unigrams and Bigrams to catch phrases like "fixed typo" or "deleted code"
    tfidf = TfidfVectorizer(
        max_features=5000, 
        stop_words='english', 
        ngram_range=(1, 2),
        sublinear_tf=True # Scalings counts logarithmically
    )

    print("Vectorizing Pre-Edit Text...")
    X_train_pre = tfidf.fit_transform(train_df['PreEditText'].fillna(""))
    X_test_pre = tfidf.transform(test_df['PreEditText'].fillna(""))

    print("Vectorizing Edited Text...")
    # Using a second vectorizer to catch vocabulary shifts in the edit
    tfidf_post = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_post = tfidf_post.fit_transform(train_df['EditedText'].fillna(""))
    X_test_post = tfidf_post.transform(test_df['EditedText'].fillna(""))

    # 3. Combine Features
    # We horizontally stack the two sparse matrices
    X_train_full = hstack([X_train_pre, X_train_post])
    X_test_full = hstack([X_test_pre, X_test_post])

    # 4. Train Model
    # Balanced class weight is crucial since Rejections are rare
    model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs')
    model.fit(X_train_full, y_train)

    # 5. Predict and Save
    probs = model.predict_proba(X_test_full)[:, 1]
    
    # Calculate Metrics
    auc = roc_auc_score(y_test, probs)
    print(f"\nTF-IDF Baseline AUC: {auc:.4f}")
    print("\nClassification Report:")
    preds = (probs > 0.5).astype(int)
    print(classification_report(y_test, preds))

    # Save results for final comparison table
    results_path = 'results/ensemble_base.csv'
    if os.path.exists(results_path):
        res_df = pd.read_csv(results_path)
        res_df['Prob_TFIDF'] = probs
        res_df.to_csv(results_path, index=False)
        print(f"TF-IDF probabilities saved to {results_path}")

    return model, tfidf

if __name__ == "__main__":
    import os
    run_tfidf_baseline()