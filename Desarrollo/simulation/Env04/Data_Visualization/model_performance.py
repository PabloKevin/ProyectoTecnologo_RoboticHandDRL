from observer_predictions import Predictor as ObserverPredictor
from actor_predictions import Predictor as ActorPredictor
import numpy as np
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score,
                             f1_score, precision_score, recall_score
                             )
import matplotlib.pyplot as plt


class Model_Metrics():
    def __init__(self, df_test, thresholds_list=[0.5, 1.6+0.5, 3.2+0.5, float("inf")], class_names_list=["agarre0", "agarre1", "agarre2", "agarre3"]):
        self.df_test = df_test
        
        self.thresholds_list = thresholds_list 
        self.class_names_list = class_names_list

        self.thresholds = self.thresholds_list[0]
        self.class_names = self.class_names_list[0]
        
        self.true_labels, self.pred_labels = self.pred_labels2classes()
        #self.binary_true_labels, self.binary_pred_labels = self.calculate_binary_labels()
        #self.pred_scores = self.calculate_predScores()
        #self.bool_true_labels = self.calculate_bool_labels()

        self.f1, self.precision_val, self.recall_val, self.accuracy = self.calculate_metrics(show=False)

    def pred_labels2classes(self):
        self.true_labels = self.get_class_from_reg(self.df_test["true_label"], self.thresholds)
        self.pred_labels = self.get_class_from_reg(self.df_test["predicted_label"], self.thresholds)
        return self.true_labels, self.pred_labels

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

    def dot_colors(self):
        colors = []
        for i in range(len(self.class_names)):
            colors.append(np.random.randint(0, 256, 3)/255)
        
        dot_color = []
        for label in self.df_test["true_label"]:
            for i,th in enumerate(self.thresholds):
                if label < th:
                    dot_color.append(colors[i])
                    break
        return colors, dot_color

    def plot_predicted_vs_true(self, ax=None):
        colors, dot_color = self.dot_colors()
        
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
        self.f1 = f1_score(self.true_labels, self.pred_labels, average="weighted")
        self.precision_val = precision_score(self.true_labels, self.pred_labels, average="weighted")
        self.recall_val = recall_score(self.true_labels, self.pred_labels, average="weighted")
        self.accuracy = np.mean(np.array(self.true_labels) == np.array(self.pred_labels))

        if show:
            print(f'F1-Score: {self.f1:.2f}')
            print(f'Precision: {self.precision_val:.2f}')
            print(f'Recall: {self.recall_val:.2f}')
            print(f'Accuracy: {self.accuracy:.2f}')

        return self.f1, self.precision_val, self.recall_val, self.accuracy
    
    def update(self):
        self.pred_labels2classes()
        #self.calculate_binary_labels()
        #self.calculate_predScores()
        #self.calculate_bool_labels()
        self.calculate_metrics(show=False)

class Observer_Metrics(Model_Metrics):
    def __init__(self, model_weight_dir, model_name, conv_channels, hidden_layers, thresholds_list, class_names_list):
        self.model_weight_dir = model_weight_dir
        self.model_name = model_name
        self.conv_channels = conv_channels
        self.hidden_layers = hidden_layers
        self.predictor = ObserverPredictor(conv_channels=conv_channels, hidden_layers=hidden_layers, model_weights_file=model_weight_dir+model_name)

        super().__init__(df_test=self.predictor.df_test, thresholds_list=thresholds_list, class_names_list=class_names_list)

    def show_model_performance(self, update=False):
        if update:
            self.update()
        #fig, axs = plt.subplots(2, 2, figsize=(15, 13))
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        self.plot_confusion_matrix(ax=axs[0])
        #self.plot_confusion_matrix(ax=axs[0, 0])
        #self.plot_predicted_vs_true(ax=axs[1, 0])
        
        metrics_text_0 = (
            "METRICS:\n"
            f"F1-Score: {self.f1:.2f}\n"
            f"Precision: {self.precision_val:.2f}\n"
            f"Recall: {self.recall_val:.2f}\n"
            f"Accuracy: {self.accuracy:.2f}"
        )
        fig.text(0.05, 0.1, metrics_text_0, fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

        #thresholds text
        fig.text(0.4, 0.1, f"thresholds:\n{self.thresholds}", fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

        self.thresholds = self.thresholds_list[1]
        self.class_names = self.class_names_list[1]
        self.update()
        self.plot_confusion_matrix(ax=axs[1])
        #self.plot_predicted_vs_true(ax=axs[1, 1])

        metrics_text_1 = (
            "METRICS:\n"
            f"F1-Score: {self.f1:.2f}\n"
            f"Precision: {self.precision_val:.2f}\n"
            f"Recall: {self.recall_val:.2f}\n"
            f"Accuracy: {self.accuracy:.2f}"
        )
        fig.text(0.58, 0.1, metrics_text_1, fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

        #thresholds text
        fig.text(0.7, 0.1, f"thresholds:\n{self.thresholds}", fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Add figure-wide title
        fig.text(0.1, 0.98, f'Model "{self.model_name}" Performance', 
                 ha='left', va='top', fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0.01, 1, 0.938], h_pad=5.0, w_pad=1.5)  # [left, bottom, right, top] Leave space at bottom for metrics

        # Draw separators
        #fig.add_artist(plt.Line2D([0, 1], [0.445, 0.445], color='grey', linewidth=1, linestyle='--'))  # horizontal
        fig.add_artist(plt.Line2D([0.5, 0.5], [0, 0.955], color='grey', linewidth=1, linestyle='--'))  # vertical

        plt.show()

class Actor_Metrics(Model_Metrics):
    def __init__(self, model_weight_dir, model_name, hidden_layers, class_names, file_name=None):
        self.model_weight_dir = model_weight_dir
        self.model_name = model_name
        self.hidden_layers = hidden_layers
        self.predictor = ActorPredictor(hidden_layers=hidden_layers, model_weights_dir=model_weight_dir, file_name=file_name)

        super().__init__(df_test=self.predictor.df_test, class_names_list=[class_names])

    def pred_labels2classes(self):
        self.true_labels = self.get_class_from_reg(self.df_test["true_agarres"])
        self.pred_labels = self.get_class_from_reg(self.df_test["predicted_agarres"])
        return self.true_labels, self.pred_labels
    
    def get_class_from_reg(self, agarres):
        agarres_lab = ["agarre_0", "agarre_1", "agarre_2", "agarre_3", "agarre_indefinido"]
        class_names = []
        for a in agarres:
            for i, agarre in enumerate(agarres_lab):
                if a == agarre:
                    class_names.append(i)
                    break
        return class_names
    
    def plot_confusion_matrix(self, ax=None):
        show = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8)) 
            show = True
        
        all_labels = list(range(len(self.class_names)))  # [0, 1, 2, 3, 4] for 5 classes
        cm = confusion_matrix(self.true_labels, self.pred_labels, labels=all_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap=plt.cm.Reds, ax=ax, colorbar=True)  # avoids creating new fig

        ax.set_title(f"Confusion Matrix")

        ax.set_xticklabels(self.class_names, rotation=45)

        if show:
            ax.set_title(f"Confusion Matrix\nmodel: {self.model_name}")
            plt.tight_layout(rect=[0, 0.01, 1, 0.955], h_pad=5.0, w_pad=1.5)  # [left, bottom, right, top] Leave space at bottom for metrics
            plt.show()

    def show_model_performance(self):
        self.plot_confusion_matrix()
        self.calculate_metrics(show=True)
        
# Example usage
if __name__ == "__main__":
    # OBSERVER PERFORMANCE

    #model_weight_dir = "Desarrollo/simulation/Env04/tmp/observer/"
    #model_name = "observer_best_test"
    #model_weight_dir = "Desarrollo/simulation/Env04/tmp/observer_backup/"
    #model_name = "observer_best_test_logits_best2"

    model_weight_dir = "Desarrollo/simulation/Env04/model_weights_docs/observer/v7/"
    model_name = "observer_final_v7"
    #model_name = "observer_epoch_90"

    conv_channels = [16, 32, 64]
    hidden_layers = [64, 32, 16]
    #conv_channels = [16, 32, 64]
    #hidden_layers = [64, 16, 8]

    #thresholds_1 = [0.05, 0.35, 0.65, float("inf")] # ideal 
    thresholds_1 = [0.5, 3.5, 6.5, float("inf")] # ideal 
    #thresholds = [0.9, 2.19+(2.57-2.19)/2, 3.201, float("inf")] #small
    #thresholds = [0.5, 2.599, 3.201, float("inf")] #big
    #thresholds = [0.5, 1.15, 1.45, 2.1, 2.75, 3.05, 3.7, 4.35, 4.65, float("inf")] # 10 classes

    class_names_0 = ["empty", "tuerca", "tornillo", "clavo", "lapicera", "tenedor", "cuchara", "destornillador", "martillo", "pinza"]
    class_names_1 = ["agarre0", "agarre1", "agarre2", "agarre3"]
    thresholds_0 = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, float("inf")]
    thresholds_list = [thresholds_0,thresholds_1]
    class_names_list = [class_names_0, class_names_1]
    #class_names = ["empty", "tuerca", "tornillo", "clavo", "lapicera", "tenedor", "cuchara", "destornillador", "martillo", "pinza"]

    observer_performance = Observer_Metrics(conv_channels=conv_channels, hidden_layers=hidden_layers, model_weight_dir=model_weight_dir, 
                                            model_name=model_name, thresholds_list=thresholds_list, class_names_list=class_names_list)
    #observer_performance.show_model_performance()


    # ACTOR PERFORMANCE

    #model_weight_dir = "Desarrollo/simulation/Env04/tmp/td3"
    #model_weight_dir = "Desarrollo/simulation/Env04/models_params_weights/td3"
    #model_name = "Actor_Last_Trained_Model"
    model_weight_dir = "Desarrollo/simulation/Env04/model_weights_docs/td3/v2_trainset"
    model_name = "actor_episode_25000"
    hidden_layers = [64,32,16]
    class_names = ["agarre_0", "agarre_1", "agarre_2", "agarre_3", "agarre_indefinido"]

    actor_performance = Actor_Metrics(hidden_layers=hidden_layers, model_weight_dir=model_weight_dir, model_name=model_name, class_names=class_names, file_name=model_name)
    actor_performance.show_model_performance()
    

   

    
