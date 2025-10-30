# explain/visualize_metrics.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(6,5))
    plt.imshow(np.array(cm), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()