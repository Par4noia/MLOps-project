import argparse
from pathlib import Path
import pandas as pd

from config import RAW_FILE, PROCESSED_FILE, QUALITY_REPORT, BATCH_SIZE
from src.collection import stream_batches, append_raw
from src.analysis import data_quality, clean_data, save_quality_report


def handle_update():
    from config import DATA_FILE
    if not RAW_FILE.exists():
        print(f"raw file not found, collecting data...")
        for batch in stream_batches(DATA_FILE, BATCH_SIZE):
            append_raw(batch, RAW_FILE)
    df = pd.read_csv(RAW_FILE)
    quality = data_quality(df)
    save_quality_report(quality, QUALITY_REPORT)
    df_clean = clean_data(df)
    df_clean.to_csv(PROCESSED_FILE, index=False)
    print(f"processed data saved to {PROCESSED_FILE}")
    print(f"quality report saved to {QUALITY_REPORT}")
    return True


def handle_inference(file):
    print("inference mode: not implemented yet")
    return None


def handle_summary():
    print("summary mode: not implemented yet")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", required=True, choices=["inference", "update", "summary"])
    parser.add_argument("-file", default=None)
    args = parser.parse_args()

    if args.mode == "update":
        ok = handle_update()
        return 0 if ok else 1

    if args.mode == "inference":
        out = handle_inference(args.file)
        return 0 if out is not None else 1

    if args.mode == "summary":
        out = handle_summary()
        return 0 if out is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())