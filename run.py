import argparse
from pathlib import Path
import pandas as pd

from config import RAW_FILE, PROCESSED_FILE, QUALITY_REPORT, BATCH_SIZE, MODEL_DIR, REPORT_DIR
from src.collection import stream_batches, append_raw
from src.analysis import data_quality, make_target, clean_data, save_quality_report
from src.preparation import preprocess_data, split_by_time
from src.training import train_decision_tree, train_neural_net, evaluate_model, save_model
from src.serving import process_inference_file
from src.summary import generate_summary_report

# Файл для сохранения метрик модели
MODEL_METRICS_FILE = REPORT_DIR / "model_metrics.json"


def handle_update():
    from config import DATA_FILE
    if not RAW_FILE.exists():
        print(f"raw file not found, collecting data...")
        for batch in stream_batches(DATA_FILE, BATCH_SIZE):
            append_raw(batch, RAW_FILE)
    df = pd.read_csv(RAW_FILE)
    quality = data_quality(df)
    save_quality_report(quality, QUALITY_REPORT)
    df = make_target(df)
    df_clean = clean_data(df)
    df_clean.to_csv(PROCESSED_FILE, index=False)
    print(f"processed data saved to {PROCESSED_FILE}")
    print(f"quality report saved to {QUALITY_REPORT}")
    df_prep = preprocess_data(df_clean)
    train_df, test_df = split_by_time(df_prep)
    cols_to_drop = ['INSR_BEGIN', 'INSR_END']
    X_train = train_df.drop(columns=['target'] + [c for c in cols_to_drop if c in train_df.columns])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'] + [c for c in cols_to_drop if c in test_df.columns])
    y_test = test_df['target']
    dt_model = train_decision_tree(X_train, y_train)
    nn_model = train_neural_net(X_train, y_train)
    dt_metrics = evaluate_model(dt_model, X_test, y_test)
    nn_metrics = evaluate_model(nn_model, X_test, y_test)
    print(f"Decision Tree: acc={dt_metrics['accuracy']:.3f}, f1={dt_metrics['f1_score']:.3f}")
    print(f"Neural Net: acc={nn_metrics['accuracy']:.3f}, f1={nn_metrics['f1_score']:.3f}")

    best_model = dt_model if dt_metrics['f1_score'] > nn_metrics['f1_score'] else nn_model
    best_metrics = dt_metrics if dt_metrics['f1_score'] > nn_metrics['f1_score'] else nn_metrics
    best_params = {
        "model_type": "DecisionTree" if dt_metrics['f1_score'] > nn_metrics['f1_score'] else "NeuralNetwork",
        "dt_params": dt_model.get_params(),
        "nn_params": nn_model.get_params()
    }

    import json
    with open(MODEL_METRICS_FILE, 'w') as f:
        json.dump({"metrics": best_metrics, "params": best_params}, f, indent=2)

    model_path = MODEL_DIR / "best_model.pkl"
    save_model(best_model, model_path)
    print(f"best model saved to {model_path}")
    return True


def handle_inference(file):
    if not file:
        print("error: -file required for inference")
        return None
    model_path = MODEL_DIR / "best_model.pkl"
    if not model_path.exists():
        print(f"model not found: {model_path}")
        return None
    output_path = Path(file).parent / f"{Path(file).stem}_predictions.csv"
    result_path = process_inference_file(file, output_path, model_path)
    print(f"predictions saved to {result_path}")
    return result_path


def handle_summary():
    import json
    model_metrics = {}
    model_params = {}

    if MODEL_METRICS_FILE.exists():
        with open(MODEL_METRICS_FILE, 'r') as f:
            data = json.load(f)
            model_metrics = data.get("metrics", {})
            model_params = data.get("params", {})

    summary_path = REPORT_DIR / "summary_report.json"
    report_path = generate_summary_report(
        quality_path=QUALITY_REPORT,
        model_metrics=model_metrics,
        model_params=model_params,
        output_path=summary_path
    )
    print(f"summary report saved to {report_path}")
    return report_path


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