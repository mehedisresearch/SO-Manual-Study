from prompt_toolkit import prompt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os


# =================================================================
# 1. LLM CONNECTOR (Qwen2-7B-Instruct)
# =================================================================

# Enable memory efficient GPU usage
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Check available GPUs
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0)}")

model_name = "Qwen/Qwen2-7B-Instruct"

print(f"Loading model {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision to save memory
    device_map="auto"  # Automatically distributes across GPUs
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

def clip_chars(s, n=10000):
    s = "" if s is None else str(s)
    s = s.strip()
    if len(s) <= n:
        return s
    return s[:n] + "\n...[TRUNCATED]..."

def call_llm(prompt, max_new_tokens=128, max_input_tokens=2048, disable_kv_cache=False):
    """
    Memory-safe generation:
    - token truncation to cap KV cache
    - inference_mode to avoid graphs
    - optional use_cache=False to reduce peak VRAM further
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that outputs strictly in JSON format."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize with truncation (this is critical)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )

    # Move to the model's first device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Optionally reduce peak VRAM by disabling KV cache
    if disable_kv_cache:
        model.config.use_cache = False

    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    # Decode only the new tokens
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

# =================================================================
# 2. PROMPT TEMPLATES
# =================================================================

def get_baseline_prompt(row):
    return f"""
    ### Role: Expert Stack Overflow Peer Reviewer.
    ### Input:
    - Comment: {row.get('EditorComment', 'N/A')}
    - Text: {row.get('Pre_Text', 'N/A')} -> {row.get('Post_Text', 'N/A')}
    - Code: {row.get('Pre_Code', 'N/A')} -> {row.get('Post_Code', 'N/A')}

    ### Task: Classify as "Accepted" or "Rejected".
    ### Rejection Categories:
    1. No Meaningful Improvement, 2. Unnecessary Formatting, 3. Unnecessary Content, 
    4. Incorrect/Risky, 5. Misuse (Reply/Question), 6. Alternate Solution, 
    7. Spam/Vandalism, 8. Conduct Violation, 9. Low Quality, 10. Gratitude/Greetings.

    ### Output Format (Strict JSON):
    {{"Decision": "Accepted/Rejected", "Category": "Category Name", "Reasoning": "..."}}
    """

def get_reflection_prompt(row):
    return f"""
    ### ROLE
    You are a senior Stack Overflow moderator performing a post-hoc audit.
    If there exists a plausible technical or editorial justification for accepting this edit, you MUST overturn the rejection.

    INPUT
    Comment:
    {clip_chars(row.get('EditorComment', 'N/A'), 800)}

    Text BEFORE:
    {clip_chars(row.get('Pre_Text', 'N/A'), 800)}

    Text AFTER:
    {clip_chars(row.get('Post_Text', 'N/A'), 800)}

    Code BEFORE:
    {clip_chars(row.get('Pre_Code', 'N/A'), 1200)}

    Code AFTER:
    {clip_chars(row.get('Post_Code', 'N/A'), 1200)}

    OUTPUT (STRICT JSON ONLY)
    {{
    "Final_Decision": "Accepted" or "Rejected",
    "Audit_Category": "Accepted_But_Risky|Accepted_Refactor|Accepted_Clarification|Accepted_Formatting|True_Rejection_No_Merit|True_Rejection_Broken",
    "Audit_Note": "One-sentence justification"
    }}
    """


# =================================================================
# 3. CORE EXECUTION ENGINE
# =================================================================



import gc
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os

class ModerationAnalyzer:
    def __init__(self, df):
        self.df = df
        self.results_baseline = []
        self.results_reflection = []
        self.checkpoint_file = '/gpuhome/shanto1/SuggestedEditClassification/SO_Edit_Classification/results/reflection_checkpoint.json'

    def load_checkpoint(self):
        """Load checkpoint if it exists."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"✓ Checkpoint loaded: Processed {len(checkpoint['predictions'])} rows")
            return checkpoint['predictions'], checkpoint['last_processed_idx']
        return [], -1

    def save_checkpoint(self, predictions, last_idx):
        """Save progress checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'predictions': predictions,
                'last_processed_idx': last_idx,
                'timestamp': pd.Timestamp.now().isoformat()
            }, f, indent=2)

    def run_baseline(self, sample_size=None, batch_size=10):
        print("\n[1/2] Running Zero-Shot Baseline...")
        data = self.df.head(sample_size) if sample_size else self.df
        
        dataset = data[['EditorComment', 'Pre_Text', 'Post_Text', 'Pre_Code', 'Post_Code']].reset_index()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            for i in range(len(batch['index'])):
                row_dict = {key: batch[key][i] for key in batch.keys() if key != 'index'}
                prompt = get_baseline_prompt(row_dict)
                response = call_llm(prompt)
                try:
                    self.results_baseline.append(json.loads(response))
                except:
                    self.results_baseline.append({"Decision": "Accepted", "Category": "Error"})
            
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Batch {batch_idx + 1} complete. Memory cleared.")

    def run_reflection_filter(self, fp_dataframe, batch_size=10):
        """Audit false positives with checkpoint resume capability."""
        print("\n[2/2] Running Reflection Audit on False Positives...")
        
        # Load existing checkpoint if available
        final_preds, last_processed_idx = self.load_checkpoint()
        
        fp_df = fp_dataframe.reset_index(drop=True)
        start_idx = last_processed_idx + 1
        
        if start_idx > 0:
            print(f"Resuming from index {start_idx} of {len(fp_df)}")
        
        # Process remaining rows
        for idx in tqdm(range(start_idx, len(fp_df)), desc="Processing reflection batches", initial=start_idx, total=len(fp_df)):
            row = fp_df.iloc[idx]
            row_dict = {
                'EditorComment': row.get('EditorComment', 'N/A'),
                'Pre_Text': row.get('Pre_Text', 'N/A'),
                'Post_Text': row.get('Post_Text', 'N/A'),
                'Pre_Code': row.get('Pre_Code', 'N/A'),
                'Post_Code': row.get('Post_Code', 'N/A')
            }
            prompt = get_reflection_prompt(row_dict)
            response = call_llm(prompt, max_new_tokens=128, max_input_tokens=4048, disable_kv_cache=True)
            try:
                res = json.loads(response)
                decision = 1 if res.get("Final_Decision") == "Rejected" else 0
                final_preds.append(decision)
            except:
                final_preds.append(1)
            
            # Save checkpoint every 5 rows (INCREASED frequency)
            if (idx + 1) % 5 == 0:
                self.save_checkpoint(final_preds, idx)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                gc.collect()
        
        # Final save
        self.save_checkpoint(final_preds, len(fp_df) - 1)
        return final_preds

# =================================================================
# 4. MAIN SCRIPT
# =================================================================

if __name__ == "__main__":
    # 1. Load Data
    df=pd.read_parquet('/gpuhome/shanto1/SuggestedEditClassification/SO_Edit_Classification/results/Final_Test_all_results_with_probs.parquet')
    print(f"Loaded {len(df)} rows")
    analyzer = ModerationAnalyzer(df)

    # 2. Create ensemble predictions
    df['Prob_Bimodal_Pure_update'] = (df['Prob_Bimodal_Pure'] >= 0.5).astype(int)
    df['Prob_SVM_baseline_update'] = (df['Prob_SVM_baseline'] >= 0.5).astype(int)
    df['Prob_XGB_Meta_update'] = (df['Prob_XGB_Meta'] >= 0.5).astype(int)

    mock_ensemble_preds = ((df['Prob_Bimodal_Pure_update'] ==1) |  (df['Prob_XGB_Meta_update'] == 1)).astype(int) 
    
    # Filter to only rows where ensemble predicted 1 (rejected)
    ensemble_rejected = df[mock_ensemble_preds == 1].copy()
    print(f"Processing {len(ensemble_rejected)} rows where ensemble predicted rejection")
    
    # 3. Run reflection with automatic checkpoint/resume
    final_refined_preds = analyzer.run_reflection_filter(ensemble_rejected, batch_size=10)

    # 4. SAVE RESPONSES TO CSV
    results_df = pd.DataFrame(analyzer.results_reflection)
    output_filename = '/gpuhome/shanto1/SuggestedEditClassification/SO_Edit_Classification/results/qwen2_baseline_detailed_results(FP only reflect).csv'
    results_df.to_csv(output_filename, index=False)
    print(f"\n[SUCCESS] Detailed LLM responses saved to: {output_filename}")

    # 5. Metrics & Visualization
    print("\n" + "="*30)
    print("FINAL REFINED REPORT")
    print("="*30)
    print(classification_report(ensemble_rejected['y_true'].iloc[:len(final_refined_preds)], final_refined_preds))

    # Confusion Matrix
    cm = confusion_matrix(ensemble_rejected['y_true'].iloc[:len(final_refined_preds)], final_refined_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title("Confusion Matrix after LLM Reflection Audit")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig('/gpuhome/shanto1/SuggestedEditClassification/SO_Edit_Classification/results/confusion_matrix.png')
    plt.show()

# # =================================================================
# # 5. BASELINE ONLY EXECUTION (For Comparison)

# if __name__ == "__main__":

    
#     # df = pd.read_csv(
#     #     '/gpuhome/shanto1/SuggestedEditClassification/SO_Edit_Classification/results/Final_Test_all_results_with_probs.csv',
#     #     engine='c',  # Use C engine (faster for large files)
#     #     encoding='utf-8',
#     #     on_bad_lines='skip',  # Skip malformed lines instead of failing
#     #     dtype={'SuggestedId': 'int64', 'PostId': 'float64', 'y_true': 'int64'},  # Specify dtypes to speed up parsing
#     #     low_memory=False  # Load entire file at once for consistent parsing
#     # )

#     df=pd.read_parquet('/gpuhome/shanto1/SuggestedEditClassification/SO_Edit_Classification/results/Final_Test_all_results_with_probs.parquet')
    
#     print(f"Loaded {len(df)} rows")
#     analyzer = ModerationAnalyzer(df)
#     # # ...existing code...

#     # # 2. Execution (Baseline Only)
#     # We only run the baseline. You can set sample_size=None for the full file.
#     analyzer.run_baseline(sample_size=None, batch_size=50) 


#     # 3. SAVE RESPONSES TO CSV
#     results_df = pd.DataFrame(analyzer.results_baseline)
#     output_filename = '/gpuhome/shanto1/SuggestedEditClassification/SO_Edit_Classification/results/qwen2_baseline_detailed_results(without Reflection).csv'
#     results_df.to_csv(output_filename, index=False)
#     print(f"\n[SUCCESS] Detailed LLM responses saved to: {output_filename}")

#     # 3. Extract Predictions from Baseline Results
#     # Since the baseline returns a list of JSON objects, we convert them to 1s and 0s
#     y_pred_baseline = [1 if r.get("Decision") == "Rejected" else 0 for r in analyzer.results_baseline]
    
#     # Slice y_true to match the sample size of the baseline results
#     y_true_subset = df['y_true'].iloc[:len(y_pred_baseline)]

#     # 4. Final Metrics
#     print("\n" + "="*30)
#     print("ZERO-SHOT BASELINE REPORT")
#     print("="*30)
#     print(classification_report(y_true_subset, y_pred_baseline))

#     # Confusion Matrix
#     cm = confusion_matrix(y_true_subset, y_pred_baseline)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') # Changed color to Blue for Baseline
#     plt.title("Confusion Matrix: Zero-Shot Baseline")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.show()
