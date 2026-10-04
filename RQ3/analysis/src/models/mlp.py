import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class StackOverflowMLP(nn.Module):
    def __init__(self, input_dim):
        super(StackOverflowMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

def train_mlp(X_train, y_train, X_val, y_val, epochs=1000, batch_size=1024, patience=15):
    """
    patience: How many epochs to wait after last time validation loss improved.
    """
    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train.values)
    y_train_t = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val.values)
    y_val_t = torch.FloatTensor(y_val.values).reshape(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackOverflowMLP(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    print(f"Starting MLP training on {device}...")
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        with torch.no_grad():
            X_val_t, y_val_t = X_val_t.to(device), y_val_t.to(device)
            val_outputs = model(X_val_t)
            avg_val_loss = criterion(val_outputs, y_val_t).item()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict() # Save best weights
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.5f}")
            model.load_state_dict(best_model_state) # Restore best weights
            break
            
    return model.to("cpu") # Return to CPU for easier prediction in main.py