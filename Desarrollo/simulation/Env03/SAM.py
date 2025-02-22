import os
import cv2
import random
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def sample_point_gaussian(width, height, std_frac=0.2):
    """
    Samples (x, y) from a 2D normal distribution centered in the image.
    std_frac is the fraction of width/height used as standard deviation.
    """
    mx, my = width / 2, height / 2         # center
    sigma_x, sigma_y = std_frac * width, std_frac * height
    
    # Sample from normal distribution
    x = int(random.gauss(mx, sigma_x))
    y = int(random.gauss(my, sigma_y))
    
    # Clamp to [0, width-1] and [0, height-1]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    #x = random.randint(0,640)
    #y = random.randint(0,640)
    #x = 2
    #y = 2
    return x, y

def find_input_point(img, umbral=500): 
    # Redimensionar la imagen a 256x256
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    
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
    
    # Convert to [0,1] if needed:
    #img_binaria = (img_binaria > 128).astype(np.uint8)

    # Add batch (B=1) dimension -> shape: (1, 256, 256)
    input_mask = img_binaria[None, ...]  # now 3D

    #print(input_mask.shape)

    cv2.imshow("First Mask", img_binaria)
    key = cv2.waitKey(1000)
    return input_mask
    """
    if kernel is not None:
        # Apply morphological closing to fill small holes and connect nearby white pixels
        kernel = np.ones((kernel,kernel), np.uint8)
        img_binaria = cv2.morphologyEx(img_binaria, cv2.MORPH_CLOSE, kernel)
    
    # Iterate through pixels (excluding borders)
    for i in range(1, img_binaria.shape[0]-1):
        for j in range(1, img_binaria.shape[1]-1):
            # Get 8-connected neighbors
            neighbors = img_binaria[i-1:i+2, j-1:j+2]
            # Count white pixels in neighborhood
            white_count = np.sum(neighbors == 255)
            # If majority of neighbors are white, make pixel white
            if white_count >= threshold_neighbors:  # Threshold of 5 out of 9 pixels
                img_binaria[i,j] = 255

    # Convertir la imagen a escala de grises
    img_binaria = cv2.cvtColor(img_binaria, cv2.COLOR_GRAY2BGR)

    # Guardar la imagen resultanteS
    cv2.imwrite(ruta_salida, img_binaria)

    print(f"Imagen procesada y guardada: {ruta_salida}")"""

# 1) Load your SAM model (ViT-B) and create predictor
sam = sam_model_registry["vit_b"](
    checkpoint="Desarrollo/simulation/Env03/models_params_weights/SAM/sam_vit_b_01ec64.pth"
).to("cuda:0")

predictor = SamPredictor(sam)

# 2) Loop over images
image_dir = "Desarrollo/simulation/Env03/DataSets/RawTools"
image_files = os.listdir(image_dir)

for image_file in image_files:
    img_path = os.path.join(image_dir, image_file)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Optionally resize to a fixed size
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

    # Set the image on the predictor
    predictor.set_image(img)

    height, width, _ = img.shape
    img_area = height * width

    max_tries = 10
    min_mask_area = 1000      # skip masks that are too small
    found_mask = None

    filtered_masks = []
    for attempt in range(max_tries):
        # 3) Pick a random point in a region around the center
        #    For example, restrict x to [width//3, 2*width//3] etc.
        
        x, y = sample_point_gaussian(640, 640)
        #x,y = [(1,1), (1,640), (640,1), (640,640), (320,1), (1,320),(640,320), (320,640)][attempt]
        #print(f"point({x},{y})")

        # Prepare single-point input
        input_point = np.array([[x, y]])  # shape (1,2)
        input_label = np.array([1])       # 1 = foreground label

        # 4) Predict with possible multiple mask outputs
        masks, scores, logits = predictor.predict(
            #point_coords=input_point,
            #point_labels=input_label,
            mask_input = find_input_point(img),
            multimask_output=True  # get multiple candidate masks
        )


        for mask in masks:
            mask_area = mask.sum()
            if mask_area > min_mask_area and mask_area < 0.6 * img_area:
                # We found a mask that is not too small nor too large
                filtered_masks.append(mask)
        
        if len(filtered_masks)>4:
            break
        print("masks",len(filtered_masks))
    # 5) If we found a mask, convert to white-on-black
    bw_mask = np.zeros((height, width), dtype=np.uint8)
    if len(filtered_masks) > 0:
        for mask in filtered_masks:
            bw_mask[mask] = 255
    else:
        # No suitable mask was found within max tries
        print(f"No mask found for {image_file} after {max_tries} random prompts.")

    bw_mask = cv2.resize(bw_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # 6) Show or save the mask
    cv2.imshow("SAM Mask", bw_mask)
    key = cv2.waitKey(1000)
    if key == 27:  # Escape key
        break

cv2.destroyAllWindows()


