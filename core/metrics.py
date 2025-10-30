# core/metrics.py
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import numpy as np

def classification_metrics(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    prfs = precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1_w),
        "f1_macro": float(f1_macro),
        "per_class_precision": prfs[0].tolist(),
        "per_class_recall": prfs[1].tolist(),
        "per_class_f1": prfs[2].tolist(),
        "support": prfs[3].tolist(),
        "confusion_matrix": cm.tolist()
    }