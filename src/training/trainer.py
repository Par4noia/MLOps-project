import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle
from pathlib import Path

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    return model

def train_neural_net(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return {'accuracy': acc, 'f1_score': f1}

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)