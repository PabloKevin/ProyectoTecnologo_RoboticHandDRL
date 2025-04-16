import sys, os

# Ruta del archivo actual
current_dir = os.path.dirname(__file__)
# Un nivel arriba (donde está networks.py)
parent_dir = os.path.join(current_dir, "..")
sys.path.append(os.path.abspath(parent_dir))

from observerRGB import ObserverNetwork, MyImageDataset
import numpy as np
import cv2
import polars as pl

class Predictor():
    def __init__(self, model=None, model_weights_file=None, conv_channels=None, hidden_layers=None):
        if model is None:
            if conv_channels is None or hidden_layers is None:
                self.observer = ObserverNetwork()
            else:
                self.observer = ObserverNetwork(conv_channels=conv_channels, hidden_layers=hidden_layers)
            self.observer.checkpoint_file = model_weights_file
            self.observer.load_model()
            self.observer.eval()
        else:
            self.observer = model
            self.observer.eval()

        # Gather all valid image paths
        self.image_dir = "Desarrollo/simulation/Env03/DataSets/TestSet_masks/"
        self.extensions = (".png")

        self.ds = MyImageDataset(self.image_dir)

        self.df_test = self.calculate_results()[0]
        self.df_results = self.calculate_results()[1]

    def update_observer_predictions(self, observer_predictor):
        


    def calculate_results(self):
        image_files = []
        image_labels = []
        predicted_labels = []
        for f in os.listdir(self.image_dir):
            if f.lower().endswith(self.extensions):
                image_files.append(f)
                label = self.ds.__get_label_from_filename__(f)
                image_labels.append(label)

                img = cv2.imread(self.image_dir+f, cv2.IMREAD_GRAYSCALE)
                img = np.expand_dims(img, axis=(0,1)) # se añade otra dimensión más para que no tire advertencia en el dropout2d que espera shape: N,C,H,W
                predicted_label = self.observer(img)
                predicted_labels.append(predicted_label.item())

        df_test = pl.DataFrame({"file_name": image_files, "true_label": image_labels, "predicted_label": predicted_labels})
        #print(df_test.filter(pl.col("true_label")==0.3).head())
        #print(df_test["true_label"].unique().to_list())

        labels = []
        mean_pred_labels = []
        std_pred_labels = []
        len_class = []
        names = ["empty", "tuerca", "tornillo", "clavo", "lapicera", "tenedor", "cuchara", "destornillador", "martillo", "pinza"]
        for label in df_test["true_label"].unique().to_list():
            clase = df_test.filter(pl.col("true_label")==label)
            #if label == -1.0:
                #print(clase.head())
            labels.append(label)
            mean_pred_labels.append(clase["predicted_label"].mean())
            std_pred_labels.append(clase["predicted_label"].std())
            len_class.append(clase.shape[0])
            #print(f'true label = {label}    ;    mean predicted_label = {clase["predicted_label"].mean()}')

        self.df_test = df_test
        self.df_results = pl.DataFrame({"class_names": names, "true_label": labels, "mean_predicted_label": mean_pred_labels, 
                                "std_predicted_label": std_pred_labels, "len_class": len_class})
        
        return df_test, self.df_results
        
        
    def show_results(self):
        print(self.df_results)

if __name__ == "__main__":
    # Load the model
    model_weight_file = "Desarrollo/simulation/Env03/tmp/observer/observer_best_test"
    #model_weight_file = "Desarrollo/simulation/Env03/tmp/observer_backup/observer_best_test_big"
    conv_channels = [16, 32, 64]
    hidden_layers = [64, 16, 8]
    #conv_channels = None # For the last model trained.
    predictor = Predictor(conv_channels=conv_channels, hidden_layers=hidden_layers, model_weights_file=model_weight_file)
    predictor.show_results()