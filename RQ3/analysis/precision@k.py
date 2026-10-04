import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

def calculate_recall_at_precision(y_true, y_probs, target_precision=0.90):
    """
    Finds the maximum recall achievable at a specific precision level.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Filter indices where precision is >= target_precision
    valid_indices = np.where(precisions >= target_precision)[0]
    
    if len(valid_indices) == 0:
        print(f"Target precision of {target_precision} was never reached.")
        return 0, None
    
    # Within those valid indices, find the one with the maximum recall
    best_idx = valid_indices[np.argmax(recalls[valid_indices])]
    
    max_recall = recalls[best_idx]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    
    print(f"--- High Precision Analysis ---")
    print(f"At {target_precision*100:.1f}% Precision:")
    print(f"  Max Recall: {max_recall*100:.2f}%")
    print(f"  Threshold: {best_threshold:.4f}")
    
    return max_recall, best_threshold

# Example Usage:
res_df = pd.read_csv('results/ensemble_base.csv')
calculate_recall_at_precision(res_df['y_true'], res_df['Prob_Bimodal_Pure'], 0.20)