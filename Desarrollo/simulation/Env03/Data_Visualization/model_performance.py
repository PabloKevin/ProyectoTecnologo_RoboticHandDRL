from observer_predictions import Predictor
import numpy as np
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score,
                             f1_score, precision_score, recall_score
                             )
import matplotlib.pyplot as plt


class Model_Metrics():
    def __init__(self, model_weight_dir=None, model_name=None, conv_channels=None, hidden_layers=None, thresholds=[0.5, 1.6+0.5, 3.2+0.5, float("inf")],
                 class_names=["agarre0", "agarre1", "agarre2", "agarre3"]):
        
        self.model_weight_dir = model_weight_dir
        self.model_name = model_name
        self.conv_channels = conv_channels
        self.hidden_layers = hidden_layers
        
        self.predictor = Predictor(conv_channels=conv_channels, hidden_layers=hidden_layers, model_weights_file=model_weight_dir+model_name)
        self.df_test = self.predictor.df_test
        
        self.thresholds = thresholds 
        self.class_names = class_names
        
        self.true_labels, self.pred_labels = self.pred_labels2classes()
        self.binary_true_labels, self.binary_pred_labels = self.calculate_binary_labels()

    def pred_labels2classes(self):
        self.true_labels = self.get_class_from_reg(self.df_test["true_label"], self.thresholds)
        self.pred_labels = self.get_class_from_reg(self.df_test["predicted_label"], self.thresholds)
        return self.true_labels, self.pred_labels

    def calculate_binary_labels(self):
        self.pred_labels2classes()
        binary_true_labels = []
        binary_pred_labels = []
        for i in range(len(np.unique(true_labels))): #-1 por el default
            true_lab = []
            pred_lab = []
            for j,label in enumerate(true_labels):
                if label == i:
                    true_lab.append(1)
                else:
                    true_lab.append(0)
                if pred_labels[j] == i:
                    pred_lab.append(1)
                else:
                    pred_lab.append(0)
            binary_true_labels.append(true_lab)
            binary_pred_labels.append(pred_lab)

        self.binary_true_labels = binary_true_labels
        self.binary_pred_labels = binary_pred_labels
        return binary_true_labels, binary_pred_labels

    def get_class_from_reg(self, reg, thresholds = [0.5, 1.6+0.5, 3.2+0.5, float("inf")]):
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

    def plot_confusion_matrix(self, y_true, y_pred, class_names, thresholds, model_name=""):
        """
        Plots a confusion matrix using the true labels and predictions.

        Parameters:
        - y_true: Ground truth labels
        - y_pred: Predicted labels
        - class_names: List of class names
        """
        #plt.figure()
        fig, ax = plt.subplots(figsize=(9, 9)) 
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Reds, ax=ax)
        plt.xticks(rotation=45)
        plt.title(f"Confusion Matrix\nmodel: {model_name}")

        info_text = f"thresholds: {thresholds}"
        plt.gcf().text(
            0.2, 0.04, 
            info_text,
            #transform=plt.gca().transAxes,        # Para usar coords relativas (0..1)
            fontsize=10,
            ha='left',
            #verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
        )
        plt.show()

    def plot_ROC_curve(self, y_true, y_pred, class_names, thresholds, model_name=""):
        # ROC Curve
        # Normalize your continuous prediction to 0–1 range for ROC
        pred_scores = np.abs(df_test["predicted_label"] - df_test["true_label"])
        pred_scores = 1 - (pred_scores / pred_scores.max())  # Closer to target class gets higher score

        bool_true_lables = np.array(true_labels) == np.array(pred_labels)

        fpr, tpr, _ = roc_curve(bool_true_lables, pred_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('ROC Curve\nmodel: {model_name}')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def plot_PresicionRecall_curve(self):
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(bool_true_lables, pred_scores)
        average_precision = average_precision_score(bool_true_lables, pred_scores)

        plt.figure(figsize=(10, 5))
        plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid()
        plt.show()

    def calculate_metrics(self):
        # Metrics Calculation
        f1 = f1_score(true_labels, pred_labels, average="macro")
        precision_val = precision_score(true_labels, pred_labels, average="macro")
        recall_val = recall_score(true_labels, pred_labels, average="macro")

        print(f'F1-Score: {f1:.2f}')
        print(f'Precision: {precision_val:.2f}')
        print(f'Recall: {recall_val:.2f}')
        print(f'Accuracy: {np.mean(np.array(true_labels) == np.array(pred_labels)):.2f}')

# Example usage
if __name__ == "__main__":
    #model_weight_dir = "Desarrollo/simulation/Env03/tmp/observer/"
    #model_name = "observer_best_test"
    model_weight_dir = "Desarrollo/simulation/Env03/tmp/observer_backup/"
    model_name = "observer_best_test_medium02"

    conv_channels = [16, 32, 64]
    hidden_layers = [64, 16, 8]
    #conv_channels = [16, 32, 64]
    #hidden_layers = [64, 16, 8]

    thresholds = [0.5, 1.6+0.5, 3.2+0.5, float("inf")] # ideal 
    #thresholds = [0.9, 2.19+(2.57-2.19)/2, 3.201, float("inf")] #small
    #thresholds = [0.5, 2.599, 3.201, float("inf")] #big
    #thresholds = [0.5, 1.15, 1.45, 2.1, 2.75, 3.05, 3.7, 4.35, 4.65, float("inf")] # 10 classes

    class_names = ["agarre0", "agarre1", "agarre2", "agarre3"]
    #class_names = ["empty", "tuerca", "tornillo", "clavo", "lapicera", "tenedor", "cuchara", "destornillador", "martillo", "pinza"]


    plot_confusion_matrix(true_labels, pred_labels, class_names, thresholds, model_name)
    
    """    
    colors = []
    for label in df_test["true_label"]:
        for i,th in enumerate(thresholds):
            if label < th:
                colors.append(["red", "blue", "green", "gray"][i])
                break
    plt.scatter(df_test["predicted_label"], df_test["true_label"], color=colors)
    plt.show()
    """
    

    

    

   

    
