import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from src.models.embeddings import extract_qwen_embeddings
from src.models.multimodal_fusion import extract_code_embeddings, BimodalFusionModel
from src.utils.metrics import evaluate_predictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    train_df = pd.read_parquet('data/processed/train_bert.parquet')
    test_df = pd.read_parquet('data/processed/test.parquet')
    val_df = pd.read_parquet('data/processed/val.parquet')

    y_train = (train_df['Label'] == 'Rejected').astype(int).values
    y_val = (val_df['Label'] == 'Rejected').astype(int).values
    y_test = (test_df['Label'] == 'Rejected').astype(int).values

    # 2. Sequential Extraction
    # xt_train = extract_qwen_embeddings(train_df, 'data/processed/train_qwen_pure.npy')
    # xt_val = extract_qwen_embeddings(val_df, 'data/processed/val_qwen_pure.npy')
    # xt_test = extract_qwen_embeddings(test_df, 'data/processed/test_qwen_pure.npy')


    xt_train = np.load('data/processed/train_qwen_pure.npy')
    xt_val = np.load('data/processed/val_qwen_pure.npy')
    xt_test = np.load('data/processed/test_qwen_pure.npy')
    
    xc_train = extract_code_embeddings(train_df, 'data/processed/train_cg_pure.npy')
    xc_val = extract_code_embeddings(val_df, 'data/processed/val_cg_pure.npy')
    xc_test = extract_code_embeddings(test_df, 'data/processed/test_cg_pure.npy')

    # 3. Training Preparation
    model = BimodalFusionModel(xt_train.shape[1], xc_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.BCELoss()

    train_ds = TensorDataset(torch.from_numpy(xt_train), torch.from_numpy(xc_train), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    # 4. Training Loop (1000 Epochs + Early Stopping)
    best_loss = float('inf')
    patience = 10
    counter = 0

    print("Training Content Fusion Model...")
    for epoch in range(1000):
        model.train()
        for t, c, l in train_loader:
            optimizer.zero_grad()
            pred = model(t.to(device), c.to(device)).squeeze()
            loss = criterion(pred, l.to(device))
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            v_pred = model(torch.from_numpy(xt_val).to(device), torch.from_numpy(xc_val).to(device)).squeeze()
            v_loss = criterion(v_pred, torch.from_numpy(y_val).to(device).float()).item()
            
        print(f"Epoch {epoch}: Val Loss {v_loss:.4f}")
        if v_loss < best_loss:
            best_loss = v_loss
            counter = 0
            torch.save(model.state_dict(), 'results/bimodal_pure_best.pt')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # 5. Final Prediction
    model.load_state_dict(torch.load('results/bimodal_pure_best.pt'))
    model.eval()
    with torch.no_grad():
        test_probs = model(torch.from_numpy(xt_test).to(device), torch.from_numpy(xc_test).to(device)).cpu().numpy().flatten()
    
    evaluate_predictions(y_test, test_probs, "Bimodal Pure (Pre/Post Streams)")

    # Save probabilities
    results_df = pd.read_csv('results/ensemble_base.csv') if os.path.exists('results/ensemble_base.csv') else pd.DataFrame({'SuggestedId': test_df['SuggestedId'], 'y_true': y_test})
    results_df['Prob_Bimodal_Pure'] = test_probs
    results_df.to_csv('results/ensemble_base.csv', index=False)

if __name__ == "__main__":
    main()