import pandas as pd


def data_quality(df):
    m = df.isna().sum()
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "missing_total": int(m.sum()),
        "missing_per_col": m.to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }


def clean_data(df):
    d = df.drop_duplicates().copy()
    d = d[d.isna().mean(axis=1) <= 0.5]
    for c in d.columns:
        if d[c].dtype.kind in "biufc":
            d[c] = d[c].fillna(d[c].median())
        else:
            m = d[c].mode()
            d[c] = d[c].fillna(m.iloc[0] if not m.empty else "")
    return d


def save_quality_report(metrics, path):
    rows = []
    rows.append({"metric": "rows", "value": metrics["rows"]})
    rows.append({"metric": "cols", "value": metrics["cols"]})
    rows.append({"metric": "missing_total", "value": metrics["missing_total"]})
    rows.append({"metric": "duplicates", "value": metrics["duplicates"]})
    for c, v in metrics["missing_per_col"].items():
        rows.append({"metric": f"missing_{c}", "value": int(v)})
    pd.DataFrame(rows).to_csv(path, index=False)
