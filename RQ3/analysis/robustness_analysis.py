import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import resample
from tqdm import tqdm

def bootstrap_metrics(y_true, y_probs, threshold=0.5, n_iterations=1000):
    """
    Calculates 95% Confidence Intervals for F1, Precision, and Recall.
    """
    stats = {'f1': [], 'precision': [], 'recall': []}
    
    y_true = np.array(y_true)
    y_preds = (y_probs >= threshold).astype(int)

    print(f"Running {n_iterations} bootstrap iterations...")
    for i in tqdm(range(n_iterations)):
        # Resample with replacement
        indices = resample(np.arange(len(y_true)), replace=True)
        
        if len(np.unique(y_true[indices])) < 2:
            continue

        stats['f1'].append(f1_score(y_true[indices], y_preds[indices]))
        stats['precision'].append(precision_score(y_true[indices], y_preds[indices]))
        stats['recall'].append(recall_score(y_true[indices], y_preds[indices]))

    print("\n--- 95% Confidence Intervals ---")
    for metric, values in stats.items():
        mean = np.mean(values)
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        print(f"{metric.upper()}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")

# Usage:
res_df = pd.read_csv('results/ensemble_base.csv')
bootstrap_metrics(res_df['y_true'], res_df['Prob_Bimodal_Pure'])