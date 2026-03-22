import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import pickle
import json
from pathlib import Path
from datetime import datetime

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    return model

def train_neural_net(X_train, y_train):
    print("Training neural network...")
    model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
    model.fit(X_train, y_train)
    print("Neural network trained.")
    return model

def cross_validate_model(trained_model, X, y):
    """Time Series Cross-Validation using a trained model"""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.base import clone
    
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        model = clone(trained_model)
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_test_cv)

        acc = accuracy_score(y_test_cv, y_pred)
        f1 = f1_score(y_test_cv, y_pred, average='weighted')
        scores.append({'accuracy': acc, 'f1_score': f1})

    accuracies = [s['accuracy'] for s in scores]
    f1_scores = [s['f1_score'] for s in scores]
    
    return {
        'mean_accuracy': sum(accuracies) / len(accuracies),
        'std_accuracy': (sum((x - sum(accuracies)/len(accuracies))**2 for x in accuracies) / len(accuracies))**0.5,
        'mean_f1': sum(f1_scores) / len(f1_scores),
        'std_f1': (sum((x - sum(f1_scores)/len(f1_scores))**2 for x in f1_scores) / len(f1_scores))**0.5,
        'fold_scores': scores
    }

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        'accuracy': acc,
        'f1_score': f1,
        'classification_report': report
    }

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_model_version(model, metrics, model_name, versions_dir):
    """Сохраняет версию модели с метриками"""
    versions_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"{model_name}_{timestamp}"

    model_path = versions_dir / f"{version_name}.pkl"
    metrics_path = versions_dir / f"{version_name}_metrics.json"

    save_model(model, model_path)

    version_info = {
        "model_name": model_name,
        "timestamp": timestamp,
        "metrics": metrics,
        "model_path": str(model_path)
    }

    with open(metrics_path, 'w') as f:
        json.dump(version_info, f, indent=2)

    return version_name, model_path