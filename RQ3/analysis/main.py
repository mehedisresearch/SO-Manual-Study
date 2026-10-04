import pandas as pd
import torch
import os
from src.features.selection_scaling import prepare_numerical_data, NUMERIC_FEATURES
from src.models.traditional_ml import run_traditional_experiment
from src.models.mlp import train_mlp
from src.models.embeddings import train_bert_embeddings, predict_bert
from src.utils.metrics import evaluate_predictions

def main():
    # --- 1. SETTINGS & PATHS ---
    #TRAIN_XGB_PATH = 'data/processed/train_xgb.parquet'
    TRAIN_BERT_PATH = '/gpuhome/shanto1/SuggestedEditClassification/SO_Edit_Classification/data/processed/train_bert.parquet' # 50k subset
    VAL_PATH = '/gpuhome/shanto1/SuggestedEditClassification/SO_Edit_Classification/data/processed/val.parquet'
    TEST_PATH = '/gpuhome/shanto1/SuggestedEditClassification/SO_Edit_Classification/data/processed/test.parquet'
    RESULTS_PATH = '/gpuhome/shanto1/SuggestedEditClassification/SO_Edit_Classification/results/ensemble_base.csv'

    # --- 2. LOAD DATA ---
    print("Loading datasets...")
    #train_xgb_df = pd.read_parquet(TRAIN_XGB_PATH)
    train_bert_df = pd.read_parquet(TRAIN_BERT_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y_test = (test_df['Label'] == 'Rejected').astype(int)

    # # --- 3. MODULE 1: TRADITIONAL ML (XGB & MLP) ---
    # print("\n--- Running Traditional ML Baselines ---")
    # X_train_scaled, X_test_scaled = prepare_numerical_data(train_xgb_df, test_df)
    # y_train_xgb = (train_xgb_df['Label'] == 'Rejected').astype(int)
    
    # # XGBoost
    # from xgboost import XGBClassifier
    # xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1)
    # xgb_model.fit(X_train_scaled, y_train_xgb)
    # xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

    # # MLP with Early Stopping
    # y_val = (val_df['Label'] == 'Rejected').astype(int)
    # # Scale validation data using same logic
    # _, X_val_scaled = prepare_numerical_data(train_xgb_df, val_df, save_scaler=False)
    
    # mlp_model = train_mlp(X_train_scaled, y_train_xgb, X_val_scaled, y_val, epochs=1000)
    # mlp_model.eval()
    # with torch.no_grad():
    #     mlp_probs = mlp_model(torch.FloatTensor(X_test_scaled.values)).numpy().flatten()

    # --- 4. MODULE 2: BERT EMBEDDINGS ---
    print("\n--- Starting Module 2: BERT Fine-Tuning (50k Subset) ---")
    # Training on the small balanced subset, validating on 2022 data
    bert_model, tokenizer = train_bert_embeddings(train_bert_df, val_df, epochs=100, patience=5)
    
    # Batch Prediction on the large test set
    print("Performing BERT inference on full test set...")
    bert_probs = predict_bert(bert_model, test_df, tokenizer, batch_size=32)

    # --- 5. SAVE & EVALUATE ---
    print("\n--- Final Step: Saving Probabilities for Ensemble ---")
    results_df = pd.DataFrame({
        'SuggestedId': test_df['SuggestedId'],
        'y_true': y_test,
        # 'Prob_XGB': xgb_probs,
        # 'Prob_MLP': mlp_probs,
        'Prob_BERT': bert_probs
    })
    
    results_df.to_csv(RESULTS_PATH, index=False)
    
    # Quick Check on BERT performance alone
    evaluate_predictions(y_test, bert_probs, model_name="BERT Text Embedding")
    
    print(f"All model probabilities saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()