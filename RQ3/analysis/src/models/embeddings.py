# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertForSequenceClassification

# from transformers import BertTokenizer, BertForSequenceClassification
# from torch.optim import AdamW

# from tqdm import tqdm
# import copy

# class SOTextDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_len=512):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, item):
#         text = str(self.texts[item])
#         label = self.labels[item]

#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt',
#         )

#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.long)
#         }

# def train_bert_embeddings(train_df, val_df, epochs=1000, batch_size=16, patience=5):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
#     # Using Raw EditedText, PreEditText, and Comment
#     def combine_raw(df):
#         return (df['EditorComment'].fillna("") + " [SEP] " + 
#                 df['PreEditText'].fillna("") + " [SEP] " + 
#                 df['EditedText'].fillna("")).values

#     train_text = combine_raw(train_df)
#     train_labels = (train_df['Label'] == 'Rejected').astype(int).values
    
#     val_text = combine_raw(val_df)
#     val_labels = (val_df['Label'] == 'Rejected').astype(int).values

#     train_ds = SOTextDataset(train_text, train_labels, tokenizer)
#     val_ds = SOTextDataset(val_text, val_labels, tokenizer)

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size)

#     model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
#     optimizer = AdamW(model.parameters(), lr=2e-5)
    
#     # Early Stopping Variables
#     best_val_loss = float('inf')
#     epochs_no_improve = 0
#     best_model_wts = copy.deepcopy(model.state_dict())

#     print(f"Starting BERT fine-tuning on {device}...")

#     for epoch in range(epochs):
#         # --- Training ---
#         model.train()
#         total_train_loss = 0
#         for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             model.zero_grad()
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         avg_train_loss = total_train_loss / len(train_loader)

#         # --- Validation ---
#         model.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#                 labels = batch['labels'].to(device)
                
#                 outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#                 total_val_loss += outputs.loss.item()

#         avg_val_loss = total_val_loss / len(val_loader)
#         print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

#         # --- Early Stopping Logic ---
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             epochs_no_improve = 0
#             best_model_wts = copy.deepcopy(model.state_dict())
#             torch.save(best_model_wts, 'results/best_bert_model.pt')
#         else:
#             epochs_no_improve += 1
#             print(f"No improvement for {epochs_no_improve} epochs.")

#         if epochs_no_improve >= patience:
#             print("Early stopping triggered.")
#             model.load_state_dict(best_model_wts)
#             break

#     return model, tokenizer


# def predict_bert(model, test_df, tokenizer, batch_size=32):
#     """
#     Performs batch inference on the full test set.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
    
#     # Replicate the raw text combination logic used in training
#     test_texts = (test_df['EditorComment'].fillna("") + " [SEP] " + 
#                   test_df['PreEditText'].fillna("") + " [SEP] " + 
#                   test_df['EditedText'].fillna("")).values
    
#     # We pass a dummy label (0) because the Dataset class expects one
#     dummy_labels = np.zeros(len(test_texts))
#     test_ds = SOTextDataset(test_texts, dummy_labels, tokenizer)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
#     all_probs = []
    
#     print(f"Starting inference on {len(test_texts)} samples...")
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="BERT Predicting"):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
            
#             outputs = model(input_ids, attention_mask=attention_mask)
            
#             # Apply Softmax to get probabilities and take the 'Rejected' class (index 1)
#             probs = torch.softmax(outputs.logits, dim=1)[:, 1]
#             all_probs.extend(probs.cpu().numpy())
            
#     return all_probs







# module 2: previous body and post body, comment (text+code)

# import torch
# from transformers import AutoModel, AutoTokenizer
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
# import numpy as np
# import os

# class TextDataset(Dataset):
#     def __init__(self, texts):
#         self.texts = texts

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, i):
#         return self.texts[i]

# def extract_qwen_embeddings(df, path, model_name="Alibaba-NLP/gte-Qwen2-7B-instruct", batch_size=2):
#     """
#     Extracts embeddings and saves them to a numpy file to avoid re-processing.
#     """
#     if os.path.exists(path):
#         print(f"Loading existing embeddings from {path}...")
#         return np.load(path)

#     print(f"Extracting embeddings using {model_name}...")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Load Tokenizer and Model in FP16 to save memory
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     model = AutoModel.from_pretrained(model_name, trust_remote_code=True, 
#                                      torch_dtype=torch.float16).to(device)
#     model.eval()

#     # Combine Raw Text Streams: Comment + PreText + EditedText
#     texts = (df['EditorComment'].fillna("") + " [SEP] " + 
#              df['PreEditText'].fillna("") + " [SEP] " + 
#              df['EditedText'].fillna("")).tolist()

#     loader = DataLoader(TextDataset(texts), batch_size=batch_size)
#     all_embeddings = []
#     # Clear memory before inference
#     import gc
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
    
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="LLM Inference"):
#             inputs = tokenizer(batch, padding=True, truncation=True, 
#                               max_length=1024, return_tensors='pt').to(device)
            
#             outputs = model(**inputs)
            
#             # Using Mean Pooling of the last hidden state
#             attention_mask = inputs['attention_mask']
#             last_hidden = outputs.last_hidden_state
#             mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
#             sum_embeddings = torch.sum(last_hidden * mask, 1)
#             sum_mask = torch.clamp(mask.sum(1), min=1e-9)
#             mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
            
#             all_embeddings.append(mean_pooled.astype(np.float32))

#     final_vecs = np.vstack(all_embeddings)
#     np.save(path, final_vecs)
#     return final_vecs

# class WideMLP(torch.nn.Module):
#     """
#     A Deep Neural Network that combines Metadata (39 dims) + Qwen Vectors (3584+ dims)
#     """
#     def __init__(self, embed_dim):
#         super(WideMLP, self).__init__()
#         input_dim = embed_dim
        
#         self.network = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 1024),
#             torch.nn.ReLU(),
#             torch.nn.BatchNorm1d(1024),
#             torch.nn.Dropout(0.3),
            
#             torch.nn.Linear(1024, 512),
#             torch.nn.ReLU(),
#             torch.nn.BatchNorm1d(512),
#             torch.nn.Dropout(0.2),
            
#             torch.nn.Linear(512, 1),
#             torch.nn.Sigmoid()
#         )

#     def forward(self, x_embed):
#         return self.network(x_embed)





#Module 3: text (previous body + post body + comment) and code (pre-edit code + post-edit code)
import torch
import numpy as np
import os
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

def extract_qwen_embeddings(df, path, model_name="Alibaba-NLP/gte-Qwen2-7B-instruct", batch_size=4):
    if os.path.exists(path):
        return np.load(path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # UPDATED: Using Pre_Text and Post_Text + Comment
    texts = ("Comment: " + df['EditorComment'].fillna("") + " [SEP] " + 
             "Original: " + df['Pre_Text'].fillna("") + " [SEP] " + 
             "Edited: " + df['Post_Text'].fillna("")).tolist()

    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Qwen 4-bit (Pure Text)"):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(model.device)
            outputs = model(**inputs)
            
            # Mean Pooling
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            all_embeddings.append((sum_embeddings / sum_mask).cpu().numpy().astype(np.float32))

    final_vecs = np.vstack(all_embeddings)
    np.save(path, final_vecs)
    
    # Force VRAM Clear
    del model, tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return final_vecs