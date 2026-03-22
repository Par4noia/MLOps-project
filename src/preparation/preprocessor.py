import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

def preprocess_data(df):
    d = df.copy()
    if 'CLAIM_PAID' in d.columns:
        d = d.drop(columns=['CLAIM_PAID'])
    if 'EFFECTIVE_YR' in d.columns:
        d['EFFECTIVE_YR'] = pd.to_numeric(d['EFFECTIVE_YR'], errors='coerce')
        d['EFFECTIVE_YR'] = d['EFFECTIVE_YR'].fillna(d['EFFECTIVE_YR'].median())
    cat_cols = ['SEX', 'TYPE_VEHICLE', 'MAKE', 'USAGE']
    for col in cat_cols:
        if col in d.columns:
            le = LabelEncoder()
            d[col] = le.fit_transform(d[col].astype(str))
    num_cols = ['INSURED_VALUE', 'PREMIUM', 'PROD_YEAR', 'SEATS_NUM', 'CARRYING_CAPACITY', 'CCM_TON', 'INSR_DURATION_DAYS', 'EFFECTIVE_YR']
    scaler = StandardScaler()
    for col in num_cols:
        if col in d.columns:
            d[col] = scaler.fit_transform(d[[col]])
    return d

def split_by_time(df, time_col='INSR_BEGIN', test_size=0.2):
    df_sorted = df.sort_values(time_col)
    split_idx = int(len(df_sorted) * (1 - test_size))
    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]
    return train, test

def save_preprocessor(prep_data, path):
    with open(path, 'wb') as f:
        pickle.dump(prep_data, f)
