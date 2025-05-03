import os
from observer_predictions import Predictor as Observer_Predictor
from networks import ActorNetwork
import numpy as np
import cv2
import polars as pl
import torch as T

class Predictor():
    def __init__(self, model=None, model_weights_dir=None, hidden_layers=None):
        if model is None:
            if hidden_layers is None or model_weights_dir is None:
                self.actor = ActorNetwork()
            else:
                self.actor = ActorNetwork(checkpoint_dir=model_weights_dir, hidden_layers=hidden_layers)
        else:
            self.actor = model
        self.actor.load_checkpoint()
        self.actor.eval()

        self.observer_df_test = pl.read_csv("Desarrollo/simulation/Env04/Data_Visualization/observer_df_test.csv")

        self.df_test = self.calculate_results()

    def update_observer_predictions(self,
                                    model_weight_dir = "Desarrollo/simulation/Env04/tmp/observer_backup/",
                                    model_name = "observer_best_test_medium02",
                                    conv_channels = [16, 32, 64],
                                    hidden_layers = [64, 16, 8]
                                    ):
        obs_predictor = Observer_Predictor(conv_channels=conv_channels, hidden_layers=hidden_layers, model_weights_file=model_weight_dir+model_name)
        self.observer_df_test = obs_predictor.df_test
        obs_predictor.save_df_test()
        print(f"Observer predictions updated with model: {model_name}")

    def label_from_filename(self, filename):
        tools = ["empty", "tuerca", "tornillo", "clavo", "lapicera", "tenedor", "cuchara", "destornillador", "martillo", "pinza"]
        agarres = [[0,0,0,0,0], [2,1,2,2,2], [2,1,1,2,2], [1,1,1,1,1]]
        for i, tool in enumerate(tools):
            if filename.startswith(tool):
                if i == 0:
                    return agarres[0]
                elif i <= 3:
                    return agarres[1]
                elif i <= 6:
                    return agarres[2]
                else:
                    return agarres[3]
                
    def calculate_true_labels(self, filenames):
        true_labels = []
        for filename in filenames:
            true_labels.append(self.label_from_filename(filename))

        return true_labels

    def predict(self, tool):
        finger_actions= []
        for f_idx in range(5):
            observation = T.tensor(np.array([tool, float(f_idx)]), dtype=T.float).to(self.actor.device)
            pred_action = self.actor(observation).item()
            finger_actions.append(pred_action + 1) # pred_action (-1, 1) +1 -> (0,2) intervals and action spaces
        return finger_actions
    
    def predAction_to_closeAction(self, predActions):
        closeActions = []
        for action in predActions:
            f_action = []
            for finger in action:
                if finger < (1 - 1/3):
                    f_action.append(0)
                elif finger < (1 + 1/3):
                    f_action.append(1)
                else:
                    f_action.append(2)
            closeActions.append(f_action)
        return closeActions

    def actions_to_agarres(self, actions):
        agarres_comb = [[0,0,0,0,0], [2,1,2,2,2], [2,1,1,2,2], [1,1,1,1,1]]
        agarres = []
        for action in actions:
            end = 0
            for i, agarre in enumerate(agarres_comb):
                if agarre == action:
                    agarres.append("agarre_"+str(i))
                    end = 0
                elif end == 3:
                    agarres.append("agarre_indefinido")
                    end = 0
                else:
                    end += 1
        return agarres

    def calculate_results(self):
        true_labels = self.calculate_true_labels(self.observer_df_test["file_name"])

        pred_labels = []
        for tool in self.observer_df_test["predicted_label"]:
            pred_labels.append(self.predict(tool))
        
        pred_close_labels = self.predAction_to_closeAction(pred_labels)
        
        true_agarres = self.actions_to_agarres(true_labels)
        pred_agarres = self.actions_to_agarres(pred_close_labels)

        df_test = pl.DataFrame({"file_name": self.observer_df_test["file_name"], 
                                "true_label": true_labels, 
                                "predicted_label": pred_labels,
                                "pred_close_labels": pred_close_labels,
                                "true_agarres": true_agarres,
                                "predicted_agarres": pred_agarres})

        return df_test

    def save_df_test(self):
        df_flat = self.df_test.with_columns([
            pl.col("true_label").map_elements(str).alias("true_label"),
            pl.col("predicted_label").map_elements(str).alias("predicted_label"),
            pl.col("pred_close_labels").map_elements(str).alias("pred_close_labels")
        ])

        df_flat.write_csv("Desarrollo/simulation/Env04/Data_Visualization/actor_df_test.csv", separator=";")

        print("df_test saved to CSV file.")
        

if __name__ == "__main__":
    # Load the model
    #model_weight_file = "Desarrollo/simulation/Env04/tmp/observer/observer_best_test"
    #model_weight_file = "Desarrollo/simulation/Env04/tmp/observer_backup/observer_best_test_big"
    #hidden_layers = [64, 16, 8]
    predictor = Predictor(model_weights_dir="Desarrollo/simulation/Env04/models_params_weights/td3", hidden_layers=[32,32])
    #predictor.update_observer_predictions()
    predictor.save_df_test()
    print(predictor.df_test.head())