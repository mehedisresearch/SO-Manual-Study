import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Definitive list of numerical features based on your dataset
NUMERIC_FEATURES = [
    'Reputation', 'Flesch', 'Fog', 'LIX', 'Kincaid', 'ARI', 'ColemanLiau', 
    'SMOG', 'DaleChall', 'UserEdits24h', 'TagPopularitySum', 'TagPopularityMean',
    'PostLengthChars', 'PostLengthTokens', 'NumTags', 'NumLinks', 'NumLinksChanged', 
    'TotalVotes', 'Score', 'CommentCount', 'AnswerCount', 'PostNumEditsBefore', 
    'PostAcceptedEditsBefore', 'PostRejectedEditsBefore', 'TitleEditLength', 
    'BodyEditLength', 'TagEditLength', 'TotalEditLength', 'DelayHours', 
    'TextChangeRatio', 'CodelineBefore', 'CodelineAfter', 'CodelineChanges',
    'CodeChangePercentage', 'UserAcceptedEditsBefore', 'UserRejectedEditsBefore', 
    'UserTotalEditsBefore', 'UserAcceptanceRate', 'UserRejectionRate'
]

def prepare_numerical_data(train_df, test_df, save_scaler=True):
    """
    Cleans missing values and scales numerical features.
    """
    # 1. Fill missing values (NaNs)
    # For most SO features, 0 is a safe neutral value for missing counts/ratios
    train_df[NUMERIC_FEATURES] = train_df[NUMERIC_FEATURES].fillna(0)
    test_df[NUMERIC_FEATURES] = test_df[NUMERIC_FEATURES].fillna(0)
    
    # 2. Scale features
    scaler = StandardScaler()
    
    print(f"Scaling {len(NUMERIC_FEATURES)} features...")
    
    # Fit only on training data to avoid data leakage
    train_scaled = scaler.fit_transform(train_df[NUMERIC_FEATURES])
    test_scaled = scaler.transform(test_df[NUMERIC_FEATURES])
    
    # Convert back to DataFrame for easier handling
    train_scaled_df = pd.DataFrame(train_scaled, columns=NUMERIC_FEATURES, index=train_df.index)
    test_scaled_df = pd.DataFrame(test_scaled, columns=NUMERIC_FEATURES, index=test_df.index)
    
    if save_scaler:
        joblib.dump(scaler, 'results/scaler.joblib')
        
    return train_scaled_df, test_scaled_df