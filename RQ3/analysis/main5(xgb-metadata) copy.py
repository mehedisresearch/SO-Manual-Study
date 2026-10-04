import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
from src.features.selection_scaling import NUMERIC_FEATURES, prepare_numerical_data
from src.utils.metrics import evaluate_predictions

# ============================================================================
# TRAINING FUNCTION - Run once to train and save the model
# ============================================================================
def train_metadata_xgboost():
    """Train XGBoost on metadata features and save the model"""
    print("\n--- Training XGBoost Metadata Model ---")
    
    # 1. Load Training Data
    train_df = pd.read_parquet('data/processed/train_bert.parquet')
    y_train = (train_df['Label'] == 'Rejected').astype(int)

    # 2. Prepare Numerical Features
    X_train_meta, _ = prepare_numerical_data(train_df, train_df)  # Only need train features

    # 3. Define and Train XGBoost
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
    model.fit(X_train_meta, y_train, verbose=100)

    # 4. Save Model
    model_path = 'results/xgb_metadata_model.joblib'
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
    
    return model

# ============================================================================
# INFERENCE FUNCTION - Use saved model to predict on new test data
# ============================================================================
def predict_with_metadata_xgboost(test_df, model_path='results/xgb_metadata_model.joblib', output_path='results/ensemble_base.csv'):
    """
    Load trained model and make predictions on new test data
    
    Args:
        test_df: Test dataframe with features
        model_path: Path to saved XGBoost model
        output_path: Path to save results
    
    Returns:
        DataFrame with predictions and probabilities
    """
    print("\n--- Predicting with Trained XGBoost Model ---")
    
    # 1. Load Trained Model
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}. Please run train_metadata_xgboost() first.")
        return None
    
    model = joblib.load(model_path)
    print(f"✅ Model loaded from {model_path}")
    
    # 2. Prepare Test Features
    _, X_test_meta = prepare_numerical_data(test_df, test_df)
    
    # 3. Make Predictions
    print(f"Making predictions on {X_test_meta.shape[0]} samples...")
    probs = model.predict_proba(X_test_meta)[:, 1]
    preds = model.predict(X_test_meta)
    
    # 4. Create Results DataFrame
    results_df = pd.DataFrame({
        'SuggestedId': test_df['SuggestedId'],
        'Prob_XGB_Meta': probs,
        'Pred_XGB_Meta': preds
    })
    
    # Add true labels if available
    if 'Label' in test_df.columns:
        y_test = (test_df['Label'] == 'Rejected').astype(int)
        results_df['y_true'] = y_test
    
    # 5. Save Results
    results_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
    
    # 6. If labels available, evaluate
    if 'Label' in test_df.columns:
        y_test = (test_df['Label'] == 'Rejected').astype(int)
        print("\n--- Evaluation Results ---")
        evaluate_predictions(y_test, probs, "XGBoost Metadata")
    
    return results_df

# ============================================================================
# MAIN - Choose between training or inference
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Train new model
        train_metadata_xgboost()
    
    else:
        # Use trained model for inference
        test_df = pd.read_parquet('/gpuhome/shanto1/SuggestedEditClassification/SO_Edit_Classification/Previous Implementation/Data/test_(may 22-23).parquet')
        predict_with_metadata_xgboost(test_df)