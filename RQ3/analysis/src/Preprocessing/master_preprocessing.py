import pandas as pd
import re
import os

def extract_code_robust(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    # Standardize triple backticks
    text = re.sub(r'(?:\s*`{3}\s*)+', '```', text)
    # Extract blocks
    code_blocks = re.findall(r'```(?:[a-zA-Z]*\n)?(.*?)(?:```|$)', text, re.DOTALL)
    html_blocks = re.findall(r'<code>(.*?)</code>', text, re.DOTALL)
    combined = " ".join(code_blocks + html_blocks)
    if not combined.strip():
        indented = re.findall(r'^(?: {4}|\t)(.*)$', text, re.MULTILINE)
        combined = " ".join(indented)
    return combined.strip()

def process_and_split(input_path, output_dir):
    print("Loading large dataset...")
    df = pd.read_parquet(input_path)
    df['CreationDate'] = pd.to_datetime(df['CreationDate'])
    
    # 1. Extraction Streams
    print("Extracting Text and Code streams (Pre & Post)...")
    df['Post_Code'] = df['EditedText'].apply(extract_code_robust)
    df['Pre_Code'] = df['PreEditText'].apply(extract_code_robust)
    df['Post_Text'] = df['EditedText'].str.replace(r'```.*?```|<code>.*?</code>', '', regex=True)
    df['Pre_Text'] = df['PreEditText'].str.replace(r'```.*?```|<code>.*?</code>', '', regex=True)
    
    # 2. Chronological Splitting
    print("Splitting data chronologically...")
    train_raw = df[df['CreationDate'].dt.year <= 2021].copy()
    val_df = df[df['CreationDate'].dt.year == 2022].copy()
    test_df = df[df['CreationDate'].dt.year == 2023].copy()
    
    # 3. Create Balanced Subsets
    rejected = train_raw[train_raw['Label'] == 'Rejected']
    accepted = train_raw[train_raw['Label'] == 'Accepted']
    
    # Subset A: Balanced Large (All Rejected + same amount of Accepted)
    # Total size: ~0.8M rows. Perfect for XGBoost.
    train_balanced_large = pd.concat([
        rejected, 
        accepted.sample(n=len(rejected), random_state=42)
    ]).sample(frac=1, random_state=42)
    
    # Subset B: Balanced Tiny (Subset of Large)
    # Total size: 50,000 rows. Perfect for fine-tuning BERT/CodeBERT.
    train_balanced_tiny = train_balanced_large.sample(n=50000, random_state=42)
    
    # 4. Save to Parquet (much faster than CSV)
    print("Saving processed files...")
    os.makedirs(output_dir, exist_ok=True)
    
    train_balanced_large.to_parquet(f"{output_dir}/train_xgb.parquet")
    train_balanced_tiny.to_parquet(f"{output_dir}/train_bert.parquet")
    val_df.to_parquet(f"{output_dir}/val.parquet")
    test_df.to_parquet(f"{output_dir}/test.parquet")
    
    print(f"Preprocessing complete. XGB train size: {len(train_balanced_large)}, BERT train size: {len(train_balanced_tiny)}")

# Usage
process_and_split('data/raw/enriched_dataset4(only body edited data).parquet', 'data/processed')