import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def generate_summary_report(quality_path, model_metrics, model_params, output_path):
    """
    Генерирует отчет summary с информацией о:
    - Качестве данных
    - Метриках модели
    - Гиперпараметрах
    - Производительности
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_quality": {},
        "model_metrics": model_metrics,
        "model_params": model_params,
        "performance": {}
    }

    if quality_path.exists():
        quality_df = pd.read_csv(quality_path)
        for _, row in quality_df.iterrows():
            report["data_quality"][row["metric"]] = row["value"]

    report["performance"] = {
        "inference_time_per_sample": "not_measured",
        "memory_usage": "not_measured",
        "model_size_mb": "not_calculated"
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return output_path

def load_quality_report(path):
    """Загружает отчет качества данных"""
    if path.exists():
        return pd.read_csv(path)
    return None