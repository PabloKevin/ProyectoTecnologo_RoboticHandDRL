import os
import cv2
import random
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import torch
import matplotlib.pyplot as plt


class Segmentator(): 
    def __init__(self, checkpoint_dir="Desarrollo/simulation/Env03/models_params_weights/SAM/sam_vit_b_01ec64.pth", output_dims=(256, 256)):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint_dir)
        self.sam.to(self.device)
        print(f"Created SAM Network on device: {self.device}")
        self.predictor = SamPredictor(self.sam)
        self.output_dims = output_dims


    def find_interesting_points(self, img, umbral=500, img_shape=(640, 640), render=True): 
        # Redimensionar la imagen a 256x256
        img = cv2.resize(img, img_shape, interpolation=cv2.INTER_AREA)
        
        # Obtener el color del fondo desde la esquina superior izquierda
        color_fondo = img[0, 0]  # Esto retorna [B, G, R]

        # Calcular la diferencia con respecto al color de fondo
        diferencia = np.sum((img - color_fondo) ** 2, axis=2)

        # Umbral para distinguir fondo de objeto (ajustar según la imagen)
        umbral = umbral

        # Crear máscara: True donde difiere del fondo (martillo), False donde es fondo
        mascara = diferencia > umbral

        # Crear la imagen binaria (0 = negro, 255 = blanco)
        img_binaria = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        img_binaria[mascara] = 255

        # np.where(mask == 255) returns two arrays: (row_indices, col_indices)
        white_y, white_x = np.where(mascara == True)
        white_y_bg, white_x_bg = np.where(mascara == False)

        if white_x.shape[0] == 0:
            return None

        parts = 3
        splitted_y = np.array_split(white_y, parts)
        splitted_x = np.array_split(white_x, parts)
        # If you want a list of (x, y) tuples:
        coords = []
        for i in range(parts):
            coords.append(np.array(list((zip(splitted_x[i], splitted_y[i])))))

        coords.append(np.array(list((zip(white_x_bg, white_y_bg)))))

        if render:
            cv2.imshow("First Mask", img_binaria)
            key = cv2.waitKey(1000)

        return coords

    def find_input_points(self, coords):
        if coords is None:
            return np.array([(0, 0)]), np.array([0])
        #background = coords[0][0]
        #background_x = background[0] - 10
        #background = np.array([background_x, background[1]])
        input_points = []
        labels = [] #np.array(0)
        for split in coords[:-1]:
            random_idx = np.random.randint(0, split.shape[0])
            input_points.append(split[random_idx]) #.reshape(1, -1))
            labels.append(np.array(1))
        
        for _ in range(2):
            for split in coords[:-1]:
                random_idx = np.random.randint(0, split.shape[0])
                input_points.append(split[random_idx]) #.reshape(1, -1))
                labels.append(np.array(1))

        for _ in range(5):
            random_idx = np.random.randint(0, coords[-1].shape[0])
            input_points.append(coords[-1][random_idx]) #.reshape(1, -1))
            labels.append(np.array(0))


        input_points = np.array(input_points)
        labels = np.array(labels) #.reshape(3, 1)
        #print(input_points.shape)
        #print(labels)
        return input_points, labels


    # Predict mask for a given color image
    def predict(self, img, internal_shape=(640, 640), render = False):
        # Optionally resize to a fixed size
        img = cv2.resize(img, internal_shape, interpolation=cv2.INTER_LINEAR)

        # Set the image on the predictor
        self.predictor.set_image(img)

        height, width = internal_shape
        img_area = height * width

        max_tries = 10
        min_mask_area = 1000      # skip masks that are too small

        interesting_points = self.find_interesting_points(img, img_shape=internal_shape, render=False)

        bw_mask = np.zeros((height, width), dtype=np.uint8)

        if interesting_points is None:
            bw_mask = cv2.resize(bw_mask, self.output_dims, interpolation=cv2.INTER_NEAREST)
            return bw_mask

        filtered_masks = []
        for attempt in range(max_tries):
            input_points, input_labels = self.find_input_points(interesting_points)

            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                #mask_input = find_input_point(img),
                multimask_output=True  
            )

            for mask in masks:
                mask_area = mask.sum()
                if mask_area > min_mask_area and mask_area < 0.3 * img_area:
                    # We found a mask that is not too small nor too large
                    filtered_masks.append(mask)
            
            if len(filtered_masks)>0:
                break
            #print("masks",len(filtered_masks))
        
        if len(filtered_masks) > 0:
            for mask in filtered_masks:
                bw_mask[mask] = 1 # Es 1 para que sirva como input del observer, pero podría ser 255
        else:
            # No suitable mask was found within max tries
            print(f"No mask found for the given image after {max_tries} random prompts.")

        bw_mask = cv2.resize(bw_mask, self.output_dims, interpolation=cv2.INTER_NEAREST)

        if render:
            plt.imshow(bw_mask.squeeze(), cmap='gray')
            plt.title('Black and White Mask')
            plt.axis('off') 
            plt.show(block=False)
            plt.pause(3)
            plt.close()

        return bw_mask


"""if __name__ == "__main__":
    # 2) Loop over images
    #image_dir = "Desarrollo/simulation/Env03/DataSets/Herramientas_Complicadas"
    image_dir = "Desarrollo/simulation/Env03/DataSets/RawTools"
    image_files = os.listdir(image_dir)
    #image_files = ["tornillo03.png"]
    segmentator = Segmentator()
    for image_file in image_files:
        # 6) Show or save the mask
        img_path = os.path.join(image_dir, image_file)
        img = cv2.imread(img_path)
        bw_mask = segmentator.predict(img)
        cv2.imshow("SAM Mask", bw_mask)
        key = cv2.waitKey(3000)
        if key == 27:  # Escape key
            break

    cv2.destroyAllWindows()
"""

