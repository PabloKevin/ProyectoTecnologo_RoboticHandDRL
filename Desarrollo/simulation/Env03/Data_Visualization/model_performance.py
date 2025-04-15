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

    def plot_confusion_matrix(self, ax=None):
        show = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9)) 
            show = True

        cm = confusion_matrix(self.true_labels, self.pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap=plt.cm.Reds, ax=ax, colorbar=True)  # avoids creating new fig

        ax.set_title(f"Confusion Matrix")
        ax.set_xticklabels(self.class_names, rotation=45)

        if show:
            ax.set_title(f"Confusion Matrix\nmodel: {self.model_name}")
            info_text = f"thresholds: {self.thresholds}"
            ax.text(
                0.5, -0.15, 
                info_text,
                transform=ax.transAxes,
                fontsize=10,
                ha='center',
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
        

    def plot_ROC_curve(self, ax=None, first_update=False):
        if first_update:
            self.calculate_predScores()
            self.calculate_bool_labels()

        fpr, tpr, _ = roc_curve(self.bool_true_labels, self.pred_scores)
        roc_auc = auc(fpr, tpr)

        show = False
        if ax is None:
            fig, ax = plt.subplots()
            show = True

        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate (Recall)')
        ax.set_title(f'ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True)
        
        if show:
            ax.set_title(f'ROC Curve\nmodel: {self.model_name}')
            plt.show()


    def plot_PresicionRecall_curve(self, ax=None, first_update=False):
        if first_update:
            self.calculate_predScores()
            self.calculate_bool_labels()

        precision, recall, _ = precision_recall_curve(self.bool_true_labels, self.pred_scores)
        average_precision = average_precision_score(self.bool_true_labels, self.pred_scores)

        show = False
        if ax is None:
            fig, ax = plt.subplots()
            show = True

        ax.plot(recall, precision, color="orange", lw=2, label=f'PR curve (AP = {average_precision:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        ax.grid(True)

        if show:
            plt.show()


    def plot_predicted_vs_true(self, ax=None):
        colors = []
        for i in range(len(self.class_names)):
            colors.append(np.random.randint(0, 256, 3)/255)
        
        dot_color = []
        for label in self.df_test["true_label"]:
            for i,th in enumerate(self.thresholds):
                if label < th:
                    dot_color.append(colors[i])
                    break
        show = False
        if ax is None:
            fig, ax = plt.subplots()
            show = True

        ax.scatter(self.df_test["predicted_label"], self.df_test["true_label"], c=dot_color, alpha=0.5)
        ax.set_title("Predicted vs True Labels")
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.grid(True)

        # Create legend handles (one per class)
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label=self.class_names[i],
                    markerfacecolor=clr, markersize=10)
            for i, clr in enumerate(colors)
        ]

        ax.legend(handles=legend_elements, loc='best')
        if show:
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
    
    def update(self):
        self.pred_labels2classes()
        self.calculate_binary_labels()
        self.calculate_predScores()
        self.calculate_bool_labels()
        self.calculate_metrics(show=False)

    def show_model_performance(self, update=False):
        if update:
            self.update()
        fig, axs = plt.subplots(2, 2, figsize=(15, 13))
        #fig.subplots_adjust(top=0.8, hspace=1.5, wspace=0.3)

        self.plot_confusion_matrix(ax=axs[0, 0])
        self.plot_ROC_curve(ax=axs[0, 1])
        self.plot_PresicionRecall_curve(ax=axs[1, 1])
        self.plot_predicted_vs_true(ax=axs[1, 0])

        # Add metrics text outside grid
        metrics_text = (
            "METRICS:\n"
            f"F1-Score: {self.f1:.2f}\n"
            f"Precision: {self.precision_val:.2f}\n"
            f"Recall: {self.recall_val:.2f}\n"
            f"Accuracy: {self.accuracy:.2f}"
        )
        
        # Add figure-wide title
        fig.text(0.1, 0.98, f'Model "{self.model_name}" Performance', 
                 ha='left', va='top', fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0.01, 1, 0.955], h_pad=5.0, w_pad=1.5)  # [left, bottom, right, top] Leave space at bottom for metrics

        # Draw separators
        if len(self.class_names)==4:
            fig.add_artist(plt.Line2D([0, 1], [0.46, 0.46], color='grey', linewidth=1, linestyle='--'))  # horizontal
            fig.add_artist(plt.Line2D([0.508, 0.508], [0, 0.955], color='grey', linewidth=1, linestyle='--'))  # vertical
            w_metrics, h_metrics = (0.09, 0.52) #metrics
        elif len(self.class_names)==10:
            fig.add_artist(plt.Line2D([0, 1], [0.447, 0.447], color='grey', linewidth=1, linestyle='--'))  # horizontal
            fig.add_artist(plt.Line2D([0.525, 0.525], [0, 0.955], color='grey', linewidth=1, linestyle='--'))  # vertical
            w_metrics, h_metrics = (0.11, 0.51) #metrics
        else:
            w_metrics, h_metrics = (0.09, 0.52) #metrics

        fig.text(w_metrics, h_metrics, metrics_text, fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

        plt.show()


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

    observer_performance.class_names = ["empty", "tuerca", "tornillo", "clavo", "lapicera", "tenedor", "cuchara", "destornillador", "martillo", "pinza"]
    observer_performance.thresholds = [0.5, 1.15, 1.45, 2.1, 2.75, 3.05, 3.7, 4.35, 4.65, float("inf")]
    observer_performance.show_model_performance(update=True)

    

    

    

   

    
