import pandas as pd
from pathlib import Path

def stream_batches(source_path: Path, batch_size: int):
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    for chunk in pd.read_csv(source_path, chunksize=batch_size):
        yield chunk

def append_raw(batch: pd.DataFrame, raw_path: Path):
    mode = "a" if raw_path.exists() else "w"
    header = not raw_path.exists()
    batch.to_csv(raw_path, index=False, mode=mode, header=header)
    return raw_path