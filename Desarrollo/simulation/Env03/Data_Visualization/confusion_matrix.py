import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

def get_class_from_reg(reg):
    """
    Convert a regression value to a class label.
    """
    if reg < 0.5:
        return 0
    elif reg < 1.5:
        return 1
    elif reg < 2.5:
        return 2
    elif reg < 3.5:
        return 3
    elif reg < 4.5:
        return 4
    else:
        return 5

# Example structure for confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots a confusion matrix using the true labels and predictions.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - class_names: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace these with your actual data
    observer_predictions = [0, 1, 2, 1, 0, 2, 1]  # Example predictions
    true_labels = [0, 1, 2, 1, 0, 1, 2]  # Example true labels
    class_names = ["Class 0", "Class 1", "Class 2"]  # Replace with your class names

    plot_confusion_matrix(true_labels, observer_predictions, class_names)