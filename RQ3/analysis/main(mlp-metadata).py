import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.features.selection_scaling import prepare_numerical_data
from src.utils.metrics import evaluate_predictions
import os

class MetadataMLP(nn.Module):
    def __init__(self, input_dim):
        super(MetadataMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def run_metadata_mlp_analysis():
    print("\n--- Running MLP Metadata-Only Analysis ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    train_df = pd.read_parquet('data/processed/train_bert.parquet')
    val_df = pd.read_parquet('data/processed/val.parquet')
    test_df = pd.read_parquet('data/processed/test.parquet')

    y_train = torch.tensor((train_df['Label'] == 'Rejected').astype(int).values).float()
    y_val = torch.tensor((val_df['Label'] == 'Rejected').astype(int).values).float()
    y_test = torch.tensor((test_df['Label'] == 'Rejected').astype(int).values).float()

    # 2. Scale Numerical Features
    # Note: MLPs are much more sensitive to scaling than XGBoost
    X_train_meta, X_test_meta = prepare_numerical_data(train_df, test_df)
    _, X_val_meta = prepare_numerical_data(train_df, val_df, save_scaler=False)

    # Convert to Tensors
    train_ds = TensorDataset(torch.FloatTensor(X_train_meta.values), y_train)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    # 3. Initialize Model
    model = MetadataMLP(X_train_meta.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # 4. Training Loop
    best_loss = float('inf')
    patience = 10
    trigger = 0

    print("Training Metadata MLP...")
    for epoch in range(100):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x).squeeze()
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_x = torch.FloatTensor(X_val_meta.values).to(device)
            v_preds = model(v_x).squeeze()
            v_loss = criterion(v_preds, y_val.to(device)).item()
            
        if v_loss < best_loss:
            best_loss = v_loss
            trigger = 0
            torch.save(model.state_dict(), 'results/mlp_metadata_best.pt')
        else:
            trigger += 1
            if trigger >= patience: break

    # 5. Inference
    model.load_state_dict(torch.load('results/mlp_metadata_best.pt'))
    model.eval()
    with torch.no_grad():
        t_x = torch.FloatTensor(X_test_meta.values).to(device)
        mlp_meta_probs = model(t_x).cpu().numpy().flatten()

    evaluate_predictions(y_test.numpy(), mlp_meta_probs, "MLP Metadata Only")

    # 6. Save for Ensemble
    results_path = 'results/ensemble_base.csv'
    res_df = pd.read_csv(results_path)
    res_df['Prob_MLP_Meta'] = mlp_meta_probs
    res_df.to_csv(results_path, index=False)
    print(f"MLP Metadata probabilities saved to {results_path}")

if __name__ == "__main__":
    run_metadata_mlp_analysis()