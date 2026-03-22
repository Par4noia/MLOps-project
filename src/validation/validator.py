import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def validate_model_quality(metrics, thresholds=None):
    """Проверяет качество модели по порогам"""
    if thresholds is None:
        thresholds = {
            'min_accuracy': 0.8,
            'min_f1_score': 0.7
        }

    acc_ok = metrics.get('accuracy', 0) >= thresholds['min_accuracy']
    f1_ok = metrics.get('f1_score', 0) >= thresholds['min_f1_score']

    return {
        'accuracy_ok': acc_ok,
        'f1_ok': f1_ok,
        'overall_ok': acc_ok and f1_ok,
        'thresholds': thresholds
    }

def compare_models(models_info):
    """Сравнивает модели и выбирает лучшую"""
    if not models_info:
        return None

    best_info = max(models_info, key=lambda x: x['metrics'].get('f1_score', 0))

    return {
        'model': best_info['model'],
        'metrics': best_info['metrics'],
        'cv_results': best_info['cv_results'],
        'type': best_info['type']
    }

def save_validation_report(validation_results, path):
    """Сохраняет отчет валидации"""
    with open(path, 'w') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)

def load_model_versions(versions_dir):
    """Загружает информацию о всех версиях моделей"""
    versions_dir = Path(versions_dir)
    if not versions_dir.exists():
        return []

    versions = []
    for metrics_file in versions_dir.glob("*_metrics.json"):
        with open(metrics_file, 'r') as f:
            version_info = json.load(f)
            versions.append(version_info)

    versions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return versions

def get_best_model_version(versions_dir, metric='f1_score'):
    """Возвращает лучшую версию модели по метрике"""
    versions = load_model_versions(versions_dir)

    if not versions:
        return None

    best_version = max(versions,
                      key=lambda x: x.get('metrics', {}).get(metric, 0))

    return best_version
