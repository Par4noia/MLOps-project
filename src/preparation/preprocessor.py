import pandas as pd


def get_target_col(df):
    for name in ["Response", "response", "target", "churn"]:
        if name in df.columns:
            return name
    return df.columns[-1]


def preprocess(df):
    target = get_target_col(df)
    y = df[target]
    X = df.drop(columns=[target])
    X = X.copy()
    num = X.select_dtypes(include=["number"]).columns
    cat = X.select_dtypes(include=["object", "category"]).columns
    for c in num:
        X[c] = X[c].fillna(X[c].median())
    if len(cat):
        X = pd.get_dummies(X, columns=list(cat), drop_first=True)
    return X, y


def save_processed(X, y, path):
    df = X.copy()
    df[y.name] = y
    df.to_csv(path, index=False)
