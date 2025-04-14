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
        self.pred_scores = self.calculate_predScores()
        self.bool_true_labels = self.calculate_bool_labels()

        self.f1, self.precision_val, self.recall_val, self.accuracy = self.calculate_metrics(show=False)

    def pred_labels2classes(self):
        self.true_labels = self.get_class_from_reg(self.df_test["true_label"], self.thresholds)
        self.pred_labels = self.get_class_from_reg(self.df_test["predicted_label"], self.thresholds)
        return self.true_labels, self.pred_labels

    def calculate_binary_labels(self, first_update=False):
        if first_update:
            self.pred_labels2classes()
        binary_true_labels = []
        binary_pred_labels = []
        for i in range(len(np.unique(self.true_labels))): #-1 por el default
            true_lab = []
            pred_lab = []
            for j,label in enumerate(self.true_labels):
                if label == i:
                    true_lab.append(1)
                else:
                    true_lab.append(0)
                if self.pred_labels[j] == i:
                    pred_lab.append(1)
                else:
                    pred_lab.append(0)
            binary_true_labels.append(true_lab)
            binary_pred_labels.append(pred_lab)

        self.binary_true_labels = binary_true_labels
        self.binary_pred_labels = binary_pred_labels
        return binary_true_labels, binary_pred_labels

    def get_class_from_reg(self, reg, thresholds=None):
        """
        Convert a regression value to a class label.
        """
        if thresholds is None:
            thresholds = self.thresholds

        class_names = []
        for r in reg:
            for i, threshold in enumerate(thresholds):
                if r < threshold:
                    class_names.append(i)
                    break
        return class_names

    def plot_confusion_matrix(self):
        """
        Plots a confusion matrix using the true labels and predictions.

        Parameters:
        - y_true: Ground truth labels
        - y_pred: Predicted labels
        - class_names: List of class names
        """
        #plt.figure()
        fig, ax = plt.subplots(figsize=(9, 9)) 
        cm = confusion_matrix(self.true_labels, self.pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap=plt.cm.Reds, ax=ax)
        plt.xticks(rotation=45)
        plt.title(f"Confusion Matrix\nmodel: {self.model_name}")

        info_text = f"thresholds: {self.thresholds}"
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

    def calculate_predScores(self):
        # Normalize your continuous prediction to 0–1 range for ROC
        pred_scores = np.abs(self.df_test["predicted_label"] - self.df_test["true_label"])
        pred_scores = 1 - (pred_scores / pred_scores.max())  # Closer to target class gets higher score
        self.pred_scores = pred_scores
        return self.pred_scores
    
    def calculate_bool_labels(self):
        self.bool_true_labels = np.array(self.true_labels) == np.array(self.pred_labels)
        return self.bool_true_labels
        
    def plot_ROC_curve(self, first_update=False):
        if first_update:
            self.calculate_predScores()
            self.calculate_bool_labels()
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.bool_true_labels, self.pred_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('ROC Curve\nmodel: {self.model_name}')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def plot_PresicionRecall_curve(self, first_update=False):
        if first_update:
            self.calculate_predScores()
            self.calculate_bool_labels()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.bool_true_labels, self.pred_scores)
        average_precision = average_precision_score(self.bool_true_labels, self.pred_scores)

        plt.figure(figsize=(10, 5))
        plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid()
        plt.show()

    def plot_predicted_vs_true(self):
        colors = []
        for i in range(len(self.df_test["true_label"])):
            colors.append(np.random.randint(0, 256, 3))
        
        dot_color = []
        for label in self.df_test["true_label"]:
            for i,th in enumerate(thresholds):
                if label < th:
                    dot_color.append(["red", "blue", "green", "gray"][i])
                    break
        plt.scatter(self.df_test["predicted_label"], self.df_test["true_label"], color=dot_color, alpha=0.6)
        plt.title(f"Predicted vs True Labels")
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.legend(loc='lower left')
        plt.grid()
        plt.show()

    def calculate_metrics(self, first_update=False, show=True):
        if first_update:
            self.calculate_predScores()
            self.calculate_bool_labels()

        # Metrics Calculation
        self.f1 = f1_score(self.true_labels, self.pred_labels, average="macro")
        self.precision_val = precision_score(self.true_labels, self.pred_labels, average="macro")
        self.recall_val = recall_score(self.true_labels, self.pred_labels, average="macro")
        self.accuracy = np.mean(np.array(self.true_labels) == np.array(self.pred_labels))

        if show:
            print(f'F1-Score: {self.f1:.2f}')
            print(f'Precision: {self.precision_val:.2f}')
            print(f'Recall: {self.recall_val:.2f}')
            print(f'Accuracy: {self.accuracy:.2f}')

        return self.f1, self.precision_val, self.recall_val, self.accuracy
    
    def show_model_performance(self):
        self.plot_confusion_matrix()
        self.plot_ROC_curve()
        self.plot_PresicionRecall_curve()
        self.plot_predicted_vs_true()
        self.calculate_metrics()

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

    observer_performance = Model_Metrics(conv_channels=conv_channels, hidden_layers=hidden_layers, model_weight_dir=model_weight_dir, model_name=model_name, thresholds=thresholds, class_names=class_names)
    observer_performance.show_model_performance()

    

    

    

   

    
