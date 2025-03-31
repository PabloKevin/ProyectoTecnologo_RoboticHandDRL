import sys, os

# Ruta del archivo actual
current_dir = os.path.dirname(__file__)
# Un nivel arriba (donde está networks.py)
parent_dir = os.path.join(current_dir, "..")
sys.path.append(os.path.abspath(parent_dir))

from networks import ObserverNetwork
import numpy as np
import cv2
import polars as pl

def __get_label_from_filename__(filename):
    """
    Convert a filename to a label based on its prefix.
    Customize this function as needed if your filenames differ.
    """
    filename_lower = filename.lower()
    if filename_lower.startswith("empty"):
        return -1.0
    elif filename_lower.startswith("tuerca"):
        return 0.0
    elif filename_lower.startswith("tornillo"):
        return 0.3
    elif filename_lower.startswith("clavo"):
        return 0.6
    elif filename_lower.startswith("lapicera"):
        return 10.0
    elif filename_lower.startswith("tenedor"):
        return 10.3
    elif filename_lower.startswith("cuchara"):
        return 10.6
    elif filename_lower.startswith("destornillador"):
        return 20.0
    elif filename_lower.startswith("martillo"):
        return 20.3
    elif filename_lower.startswith("pinza"):
        return 20.6
    else:
        # Default or unknown label
        return 99.9
        



conv_channels = [4, 8, 16]
hidden_layers = [32, 8]
learning_rate = 0.0008

observer = ObserverNetwork(conv_channels=conv_channels, hidden_layers=hidden_layers, learning_rate=learning_rate)
observer.load_model()
observer.eval()

# Gather all valid image paths
image_dir = "Desarrollo/simulation/Env03/DataSets/TestSet_masks/"
extensions = (".png")
image_files = []
image_labels = []
predicted_labels = []
for f in os.listdir(image_dir):
    if f.lower().endswith(extensions):
        image_files.append(f)
        label = __get_label_from_filename__(f)
        image_labels.append(label)

        img = cv2.imread(image_dir+f, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=(0,1)) # se añade otra dimensión más para que no tire advertencia en el dropout2d que espera shape: N,C,H,W
        predicted_label = observer(img)
        predicted_labels.append(predicted_label.item())

df_test = pl.DataFrame({"file_name": image_files, "true_label": image_labels, "predicted_label": predicted_labels})
#print(df_test.filter(pl.col("true_label")==0.3).head())
#print(df_test["true_label"].unique().to_list())

for label in df_test["true_label"].unique().to_list():
    clase = df_test.filter(pl.col("true_label")==label)
    #print(clase.head())
    print(f'true label = {label}    ;    mean predicted_label = {clase["predicted_label"].mean()}')