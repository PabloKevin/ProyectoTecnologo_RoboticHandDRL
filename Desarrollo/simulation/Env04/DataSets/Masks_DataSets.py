import sys, os

# Ruta del archivo actual
current_dir = os.path.dirname(__file__)
# Un nivel arriba (donde está SAM_pipe.py)
parent_dir = os.path.join(current_dir, "..")
sys.path.append(os.path.abspath(parent_dir))

from SAM_pipe import Segmentator

import cv2
import numpy as np
import time

def RGB_to_mask(imgs_path, segmentator, output_path):
    images_list = os.listdir(imgs_path)
    os.makedirs(output_path, exist_ok=True)
    for img_name in images_list:
        img_path = os.path.join(imgs_path, img_name)
        img = cv2.imread(img_path)
        bw_mask = segmentator.predict(img, render=False)
        #bw_mask = np.expand_dims(bw_mask, axis=0)
        ruta_salida = os.path.join(output_path, f"{img_name[:-4]}_mask.png")
        # Guardar la imagen
        success = cv2.imwrite(ruta_salida, bw_mask)
        if success:
            print(f"Imagen guardada en {ruta_salida}")
        else:
            print(f"No se pudo guardar la imagen en {ruta_salida}")


if __name__ == "__main__":
    checkpoint_dir="Desarrollo/simulation/Env03/models_params_weights/"
    segmentator = Segmentator(checkpoint_dir=checkpoint_dir+"SAM/sam_vit_b_01ec64.pth")
    # TrainSet
    start_train_time = time.time()
    RGB_to_mask("Desarrollo/simulation/Env03/DataSets/TrainSet/", 
                segmentator, 
                "Desarrollo/simulation/Env03/DataSets/TrainSet_masks/")
    end_train_time = time.time()
    # TestSet
    RGB_to_mask("Desarrollo/simulation/Env03/DataSets/TestSet/", 
                segmentator, 
                "Desarrollo/simulation/Env03/DataSets/TestSet_masks/")
    end_test_time = time.time()

    print("Finished")
    print(f"Tiempo total para TrainSet: {(end_train_time - start_train_time)/60} minutos")
    print(f"Tiempo total para TestSet: {(end_test_time - end_train_time)/60} minutos")
    print(f"Tiempo promedio de segmentación por imagen: {(end_test_time - end_train_time)/len(os.listdir('Desarrollo/simulation/Env03/DataSets/TestSet/'))} segundos")