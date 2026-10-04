from xgboost import XGBClassifier
import pandas as pd
import pandas as pd
from xgboost import XGBClassifier
import joblib

def train_baseline(X_train, y_train, X_val, y_val):
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        tree_method='hist', # Fast training on CPU
        early_stopping_rounds=50,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    return model




def run_traditional_experiment(train_path, test_path, feature_cols):
    # 1. Load data
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    
    # 2. Separate X and y
    X_train = train[feature_cols]
    y_train = (train['Label'] == 'Rejected').astype(int)
    X_test = test[feature_cols]
    y_test = (test['Label'] == 'Rejected').astype(int)
    
    # 3. Train Model
    print("Training XGBoost Baseline...")
    model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 4. Get Probabilities
    probs = model.predict_proba(X_test)[:, 1]
    
    # 5. Save Model
    joblib.dump(model, 'results/traditional_model.joblib')
    
    return y_test, probs