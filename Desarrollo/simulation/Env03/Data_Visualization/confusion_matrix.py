from observer_predictions import Predictor
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

def get_class_from_reg(reg, thresholds = [0.5, 1.6+0.5, 3.2+0.5, float("inf")]):
    """
    Convert a regression value to a class label.
    """
    class_names = []
    for r in reg:
        for i, threshold in enumerate(thresholds):
            if r < threshold:
                class_names.append(i)
                break
    return class_names
        


# Example structure for confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, thresholds, model_name=""):
    """
    Plots a confusion matrix using the true labels and predictions.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - class_names: List of class names
    """
    #plt.figure()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Reds)
    plt.title(f"Confusion Matrix\nmodel: {model_name}")

    info_text = f"thresholds: {thresholds}"
    plt.gcf().text(
        0.8, 0.02, 
        info_text,
        #transform=plt.gca().transAxes,        # Para usar coords relativas (0..1)
        fontsize=10,
        ha='center',
        #verticalalignment='top', 
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )

    plt.show()

# Example usage
if __name__ == "__main__":
    model_weight_dir = "Desarrollo/simulation/Env03/tmp/observer_backup/"
    model_name = "observer_best_test_big"
    conv_channels = [32, 64, 128]
    hidden_layers = [256, 128, 64]
    predictor = Predictor(conv_channels=conv_channels, hidden_layers=hidden_layers, model_weights_file=model_weight_dir+model_name)
    df_test = predictor.df_test
    #thresholds = [0.5, 1.6+0.5, 3.2+0.5, float("inf")] # ideal 
    #thresholds = [0.9, 2.19+(2.57-2.19)/2, 3.201, float("inf")] #small
    thresholds = [0.5, 2.599, 3.201, float("inf")] #big
    true_labels = get_class_from_reg(df_test["true_label"], thresholds)
    pred_labels = get_class_from_reg(df_test["predicted_label"], thresholds)
    class_names = ["agarre0", "agarre1", "agarre2", "agarre3"]
    #class_names = ["empty", "tuerca", "tornillo", "clavo", "lapicera", "tenedor", "cuchara", "destornillador", "martillo", "pinza"]


    plot_confusion_matrix(true_labels, pred_labels, class_names, thresholds, model_name)
    
"""    colors = []
    for label in df_test["true_label"]:
        for i,th in enumerate(thresholds):
            if label < th:
                colors.append(["red", "blue", "green", "gray"][i])
                break
    plt.scatter(df_test["predicted_label"], df_test["true_label"], color=colors)
    plt.show()"""