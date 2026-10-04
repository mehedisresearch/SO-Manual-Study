import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from src.models.multimodal_fusion import ContrastiveBimodalModel
from src.models.contrastive_loss import SupConLoss
from src.utils.metrics import evaluate_predictions
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. LOAD PRE-EXTRACTED VECTORS
    # Ensure you use the "Pure" streams as decided
    xt_train = np.load('data/processed/train_qwen_pure.npy')
    xc_train = np.load('data/processed/train_cg_pure.npy')
    y_train = pd.read_parquet('data/processed/train_bert.parquet')['Label'].apply(lambda x: 1 if x=='Rejected' else 0).values

    xt_test = np.load('data/processed/test_qwen_pure.npy')
    xc_test = np.load('data/processed/test_cg_pure.npy')
    y_test = pd.read_parquet('data/processed/test.parquet')['Label'].apply(lambda x: 1 if x=='Rejected' else 0).values

    # 2. PREPARE DATASET
    train_ds = TensorDataset(torch.from_numpy(xt_train), torch.from_numpy(xc_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    # 3. INITIALIZE MODELS & LOSS
    model = ContrastiveBimodalModel(xt_train.shape[1], xc_train.shape[1]).to(device)
    con_criterion = SupConLoss(temperature=0.1)
    bce_criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 4. JOINT TRAINING LOOP
    print("Starting Supervised Contrastive Training...")
    for epoch in range(100):
        model.train()
        epoch_con_loss = 0
        epoch_bce_loss = 0
        
        for t, c, l in train_loader:
            t, c, l = t.to(device), c.to(device), l.to(device)
            optimizer.zero_grad()
            
            # Forward Pass
            z = model(t, c, return_feat=True)      # Get embedding for SupCon
            probs = model.classifier(z).squeeze() # Get prediction for BCE
            
            # Compute Losses
            loss_con = con_criterion(z, l)
            loss_bce = bce_criterion(probs, l.float())
            
            # Combine (you can weigh these: alpha*con + beta*bce)
            total_loss = loss_con + loss_bce
            
            total_loss.backward()
            optimizer.step()
            
            epoch_con_loss += loss_con.item()
            epoch_bce_loss += loss_bce.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | ConLoss: {epoch_con_loss/len(train_loader):.4f} | BCELoss: {epoch_bce_loss/len(train_loader):.4f}")

        torch.save(model.state_dict(), 'results/contrastive_model.pt')


    # 5. FINAL EVALUATION
    model.load_state_dict(torch.load('results/contrastive_model.pt'))
    model.eval()
    with torch.no_grad():
        t_test = torch.from_numpy(xt_test).to(device)
        c_test = torch.from_numpy(xc_test).to(device)
        final_probs = model(t_test, c_test).cpu().numpy().flatten()

    evaluate_predictions(y_test, final_probs, "Bimodal Fusion + Supervised Contrastive")
     # Save probabilities
    results_df = pd.read_csv('results/ensemble_base.csv')
    results_df['Prob_Contrastive'] = final_probs
    results_df.to_csv('results/ensemble_base.csv', index=False)


if __name__ == "__main__":
    main()