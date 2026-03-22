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


def make_target(df):
    d = df.copy()
    d['target'] = d['CLAIM_PAID'].notna().astype(int)
    return d


def clean_data(df):
    d = df.drop_duplicates().copy()
    d = d[d.isna().mean(axis=1) <= 0.5]
    if 'INSR_BEGIN' in d.columns:
        d['INSR_BEGIN'] = pd.to_datetime(d['INSR_BEGIN'], errors='coerce')
    if 'INSR_END' in d.columns:
        d['INSR_END'] = pd.to_datetime(d['INSR_END'], errors='coerce')
    if 'INSR_BEGIN' in d.columns and 'INSR_END' in d.columns:
        d['INSR_DURATION_DAYS'] = (d['INSR_END'] - d['INSR_BEGIN']).dt.days
    if 'OBJECT_ID' in d.columns:
        d = d.drop(columns=['OBJECT_ID'])
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
