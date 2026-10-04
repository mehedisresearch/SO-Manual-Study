import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
from src.features.selection_scaling import NUMERIC_FEATURES, prepare_numerical_data
from src.utils.metrics import evaluate_predictions

def run_metadata_xgboost_analysis():
    print("\n--- Running XGBoost Metadata-Only Analysis ---")
    
    # 1. Load Data
    # We use the same train_bert (50k) and test (400k) split for consistency
    train_df = pd.read_parquet('data/processed/train_bert.parquet')
    test_df = pd.read_parquet('data/processed/test.parquet')

    y_train = (train_df['Label'] == 'Rejected').astype(int)
    y_test = (test_df['Label'] == 'Rejected').astype(int)

    # 2. Scale and Prepare Numerical Features
    # This uses the helper you built in Module 1
    X_train_meta, X_test_meta = prepare_numerical_data(train_df, test_df)

    # 3. Define and Train XGBoost
    # We use scale_pos_weight to handle the class imbalance (Accepted vs Rejected)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        n_jobs=-1,
        random_state=42,
        eval_metric='aucpr'
    )

    print(f"Training on {X_train_meta.shape[1]} metadata features...")
    model.fit(
        X_train_meta, y_train,
        eval_set=[(X_test_meta, y_test)],
        verbose=100
    )

    # 4. Predict
    probs = model.predict_proba(X_test_meta)[:, 1]
    
    # 5. Evaluate
    print("\n--- Metadata-Only Results ---")
    evaluate_predictions(y_test, probs, "XGBoost Metadata Baseline")

    # 6. Feature Importance (Research Insight)
    importance = pd.DataFrame({
        'feature': NUMERIC_FEATURES,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    print("\nTop 10 Metadata Features:")
    print(importance.head(10))

    # 7. Save for Ensemble
    results_path = 'results/ensemble_base.csv'
    if os.path.exists(results_path):
        res_df = pd.read_csv(results_path)
    else:
        res_df = pd.DataFrame({'SuggestedId': test_df['SuggestedId'], 'y_true': y_test})
    
    res_df['Prob_XGB_Meta'] = probs
    res_df.to_csv(results_path, index=False)
    
    # Save the model
    joblib.dump(model, 'results/xgb_metadata_model.joblib')
    print(f"\nMetadata probabilities saved to {results_path}")

if __name__ == "__main__":
    run_metadata_xgboost_analysis()