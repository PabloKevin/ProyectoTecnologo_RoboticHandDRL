from observer_predictions import Predictor as ObserverPredictor
from actor_predictions import Predictor as ActorPredictor
from model_performance import Model_Metrics, Actor_Metrics
import numpy as np
import matplotlib.pyplot as plt
import os

class Observer_Metrics(Model_Metrics):
    def __init__(self, model_weight_dir, model_name, conv_channels, hidden_layers, thresholds_list, class_names_list):
        self.model_weight_dir = model_weight_dir
        self.model_name = model_name
        self.conv_channels = conv_channels
        self.hidden_layers = hidden_layers
        self.predictor = ObserverPredictor(conv_channels=conv_channels, hidden_layers=hidden_layers, model_weights_file=model_weight_dir+model_name)

        super().__init__(df_test=self.predictor.df_test, thresholds_list=thresholds_list, class_names_list=class_names_list)

    def show_model_performance(self, model_name, update=False, save_path=None, show=True):
        if update:
            self.update()
        #fig, axs = plt.subplots(2, 2, figsize=(15, 13))
        fig, axs = plt.subplots(1, 1, figsize=(13, 10))
        
        self.plot_confusion_matrix(ax=axs)
        #self.plot_confusion_matrix(ax=axs[0, 0])
        #self.plot_predicted_vs_true(ax=axs[1, 0])
        
        metrics_text_0 = (
            "METRICS:\n"
            f"F1-Score: {self.f1:.2f}\n"
            f"Precision: {self.precision_val:.2f}\n"
            f"Recall: {self.recall_val:.2f}\n"
            f"Accuracy: {self.accuracy:.2f}"
        )
        #fig.text(0.17, 0.08, metrics_text_0, fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        #fig.text(0.86, 0.06, metrics_text_0, fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        fig.text(0.84, 0.93, metrics_text_0, fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        #thresholds text
        #fig.text(0.35, 0.045, f"thresholds:\n{self.thresholds}", fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        #fig.text(0.685, 0.023, f"thresholds:\n{self.thresholds}", fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        fig.text(0.54, 0.92, f"thresholds:\n{self.thresholds}", fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Add figure-wide title
        fig.text(0.35, 0.98, f'Model Performance: {model_name}', 
                 ha='left', va='top', fontsize=14, fontweight='bold')
        
        #plt.tight_layout(rect=[0, 0.01, 1, 0.938], h_pad=5.0, w_pad=1.5)  # [left, bottom, right, top] Leave space at bottom for metrics
        plt.tight_layout(rect=[0, 0.01, 1, 0.9], h_pad=5.0, w_pad=1.5)  # [left, bottom, right, top] Leave space at bottom for metrics

        if save_path is not None:
            fig.savefig(save_path,
                        dpi=300,
                        bbox_inches='tight')

        if show:
            plt.show()


# Example usage
if __name__ == "__main__":
    # OBSERVER PERFORMANCE
    model_weight_dir = "Desarrollo/simulation/Env04/model_weights_docs/observer/v7/"
    for model_name in os.listdir(model_weight_dir):
        
        #model_name = "observer_final_v7"
        #model_name = "observer_epoch_90"

        conv_channels = [16, 32, 64]
        hidden_layers = [64, 32, 16]

        class_names_0 = ["empty", "tuerca", "tornillo", "clavo", "lapicera", "tenedor", "cuchara", "destornillador", "martillo", "pinza"]
        thresholds_0 = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, float("inf")]
        thresholds_list = [thresholds_0]
        class_names_list = [class_names_0]

        observer_performance = Observer_Metrics(conv_channels=conv_channels, hidden_layers=hidden_layers, model_weight_dir=model_weight_dir, 
                                                model_name=model_name, thresholds_list=thresholds_list, class_names_list=class_names_list)
        save_dir = "Desarrollo/Documentacion/observer/confusion_matrices/"
        name, _, epoch = model_name.split("_")
        observer_performance.show_model_performance(f'"{name}" - Epoch: {epoch}',save_path=save_dir + model_name, show=False)


    """# ACTOR PERFORMANCE

    #model_weight_dir = "Desarrollo/simulation/Env04/tmp/td3"
    #model_weight_dir = "Desarrollo/simulation/Env04/models_params_weights/td3"
    #model_name = "Actor_Last_Trained_Model"
    model_weight_dir = "Desarrollo/simulation/Env04/model_weights_docs/td3/v1_fullset"
    model_name = "actor_episode_12000"
    hidden_layers = [64,32,16]
    class_names = ["agarre_0", "agarre_1", "agarre_2", "agarre_3", "agarre_indefinido"]

    actor_performance = Actor_Metrics(hidden_layers=hidden_layers, model_weight_dir=model_weight_dir, model_name=model_name, class_names=class_names, file_name=model_name)
    actor_performance.show_model_performance()"""
    
