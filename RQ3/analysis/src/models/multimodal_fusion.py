import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import os
from tqdm import tqdm

class BimodalFusionModel(nn.Module):
    def __init__(self, text_dim, code_dim):
        super().__init__()
        # Parallel branches to process different embedding sizes
        self.text_net = nn.Sequential(nn.Linear(text_dim, 512), nn.ReLU(), nn.BatchNorm1d(512))
        self.code_net = nn.Sequential(nn.Linear(code_dim, 512), nn.ReLU(), nn.BatchNorm1d(512))
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, t, c):
        t_feat = self.text_net(t)
        c_feat = self.code_net(c)
        return self.classifier(torch.cat([t_feat, c_feat], dim=1))

def extract_code_embeddings(df, path, model_name="google/codegemma-7b", batch_size=4):
    if os.path.exists(path):
        return np.load(path)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")

    # Using the code-specific streams
    code_data = ("Before: " + df['Pre_Code'].fillna("") + "\nAfter: " + df['Post_Code'].fillna("")).tolist()
    
    all_vecs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(code_data), batch_size), desc="CodeGemma 4-bit"):
            batch = code_data[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(model.device)
            outputs = model(**inputs)
            
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            mean_pooled = (torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9))
            all_vecs.append(mean_pooled.cpu().numpy().astype(np.float32))

    final_vecs = np.vstack(all_vecs)
    np.save(path, final_vecs)
    
    del model, tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return final_vecs


class ContrastiveBimodalModel(nn.Module):
    def __init__(self, text_dim, code_dim):
        super().__init__()
        # Text Encoder
        self.text_net = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        # Code Encoder
        self.code_net = nn.Sequential(
            nn.Linear(code_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        # Projection Head (for Contrastive Loss)
        self.projection_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Classification Head (for BCE Loss)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, t, c, return_feat=False):
        t_feat = self.text_net(t)
        c_feat = self.code_net(c)
        combined = torch.cat([t_feat, c_feat], dim=1)
        
        z = self.projection_head(combined) # Contrastive vector
        
        if return_feat:
            return z
        
        return self.classifier(z)