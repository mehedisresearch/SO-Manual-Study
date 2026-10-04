import pandas as pd
import numpy as np
import torch
import os
from src.features.selection_scaling import prepare_numerical_data, NUMERIC_FEATURES
from src.models.embeddings import extract_qwen_embeddings, WideMLP
from src.utils.metrics import evaluate_predictions
from xgboost import XGBClassifier
from torch.utils.data import DataLoader, TensorDataset

def main():
    # 1. PATHS
    TRAIN_BERT_PATH = 'data/processed/train_bert.parquet' # 50k subset
    VAL_PATH = 'data/processed/val.parquet'
    TEST_PATH = 'data/processed/test.parquet'
    
    # 2. LOAD DATA
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_BERT_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Labels
    y_train = (train_df['Label'] == 'Rejected').astype(int).values
    y_val = (val_df['Label'] == 'Rejected').astype(int).values
    y_test = (test_df['Label'] == 'Rejected').astype(int).values

    # 3. NUMERICAL SCALING
    print("Scaling numerical features...")
    X_train_meta, X_test_meta = prepare_numerical_data(train_df, test_df)
    _, X_val_meta = prepare_numerical_data(train_df, val_df, save_scaler=False)

    # 4. LLM FEATURE EXTRACTION (QWEN)
    # # Note: These are saved to disk as .npy
    # X_train_embed = extract_qwen_embeddings(train_df, 'data/processed/train_qwen.npy')
    # X_val_embed = extract_qwen_embeddings(val_df, 'data/processed/val_qwen.npy')
    # X_test_embed = extract_qwen_embeddings(test_df, 'data/processed/test_qwen.npy')
    #import manually to avoid re-running every time
    X_train_embed = np.load('data/processed/train_qwen.npy')
    X_val_embed = np.load('data/processed/val_qwen.npy')
    X_test_embed = np.load('data/processed/test_qwen.npy')



    # # 5. XGBOOST BASELINE (Metadata Only)
    # print("\n--- Training XGBoost (Metadata Only) ---")
    # xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1)
    # xgb.fit(X_train_meta, y_train)
    # xgb_probs = xgb.predict_proba(X_test_meta)[:, 1]
    # evaluate_predictions(y_test, xgb_probs, "XGBoost Metadata")

    # 6. WIDE MLP TRAINING (Metadata + Qwen Embeddings)
    # print("\n--- Training Wide MLP (Metadata + Qwen) ---")
    print("\n--- Training Wide MLP (Qwen) ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # # Prep Tensors
    # train_ds = TensorDataset(torch.FloatTensor(X_train_meta.values), torch.FloatTensor(X_train_embed), torch.FloatTensor(y_train))
    # val_ds = TensorDataset(torch.FloatTensor(X_val_meta.values), torch.FloatTensor(X_val_embed), torch.FloatTensor(y_val))
    
    # Prep Tensors
    train_ds = TensorDataset(torch.FloatTensor(X_train_embed), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val_embed), torch.FloatTensor(y_val))
    

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    
    # model = WideMLP(meta_dim=len(NUMERIC_FEATURES), embed_dim=X_train_embed.shape[1]).to(device)

    model = WideMLP(embed_dim=X_train_embed.shape[1]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    # Simple Training Loop with Early Stopping
    best_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(100):
        model.train()
        for e_batch, l_batch in train_loader:
            e_batch, l_batch = e_batch.to(device), l_batch.to(device)
            optimizer.zero_grad()
            preds = model(e_batch).squeeze()
            loss = criterion(preds, l_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            # v_m, v_e, v_l = torch.FloatTensor(X_val_meta.values).to(device), torch.FloatTensor(X_val_embed).to(device), torch.FloatTensor(y_val).to(device)
            v_e, v_l =  torch.FloatTensor(X_val_embed).to(device), torch.FloatTensor(y_val).to(device)

            val_preds = model(v_e)
            val_loss = criterion(val_preds.view(-1), v_l).item()

        print(f"Epoch {epoch+1}: Val Loss {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'results/wide_mlp_best.pt')
        else:
            counter += 1
            if counter >= patience: break

    # 7. FINAL INFERENCE
    model.load_state_dict(torch.load('results/wide_mlp_best.pt'))
    model.eval()
    with torch.no_grad():
        # Large test sets should be predicted in batches to avoid OOM
        test_e = torch.FloatTensor(X_test_embed).to(device)
        mlp_probs = model(test_e).cpu().numpy().flatten()

    evaluate_predictions(y_test, mlp_probs, "Wide MLP (Qwen)")

    # 8. SAVE ENSEMBLE BASE
    results_df = pd.DataFrame({
        'SuggestedId': test_df['SuggestedId'],
        'y_true': y_test,
        # 'Prob_XGB': xgb_probs,
        'Prob_WideMLP': mlp_probs
    })
    results_df.to_csv('results/ensemble_base.csv', index=False)
    print("Done! Probabilities saved to results/ensemble_base.csv")
    

if __name__ == "__main__":
    main()