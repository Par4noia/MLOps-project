from sklearn.metrics import classification_report, accuracy_score


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }
