import argparse
from pathlib import Path

from config import DATA_FILE, RAW_FILE, BATCH_SIZE
from src.collection import stream_batches, append_raw


def handle_update():
    if not DATA_FILE.exists():
        print(f"data file not found: {DATA_FILE}")
        return False
    for batch in stream_batches(DATA_FILE, BATCH_SIZE):
        append_raw(batch, RAW_FILE)
    print(f"update done, raw data at {RAW_FILE}")
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
