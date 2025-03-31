import cv2
import numpy as np
import os

class DataAugmentator:
    def __init__(self, in_image_path, out_image_path, img_shape=(640,640,3)):
        # Directorio de imágenes originales
        self.directorio_imagenes = "Desarrollo/simulation/Env03/DataSets/"+in_image_path

        # Directorio para guardar las imágenes procesadas
        self.directorio_salida = "Desarrollo/simulation/Env03/DataSets/"+out_image_path

        self.img_shape = img_shape
        # Asegurarse de que el directorio de salida existe y crearlo si no existe
        os.makedirs(self.directorio_salida, exist_ok=True)


    def scale_image(self, img, min_scale=0.9, max_scale=1.2):
        """
        Scale the image randomly within the specified range, ensuring objects don't get cut off.
        """
        # Get the dimensions of the image
        height, width, _ = self.img_shape  # Unpack the third value for color channels

        # Randomly choose a scaling factor
        scale_factor = np.random.uniform(min_scale, max_scale)

        # Calculate the new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image
        scaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # If the image is scaled down, pad it to the original size
        if scale_factor < 1.0:
            pad_x = (width - new_width) // 2
            pad_y = (height - new_height) // 2
            border_color =  tuple(int(c) for c in img[0, 0]) 
            scaled_img = cv2.copyMakeBorder(scaled_img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=border_color)
            scaled_img = scaled_img[:height, :width]  # Ensure the size matches exactly

        # If the image is scaled up, crop it to the original size
        elif scale_factor > 1.0:
            start_x = (new_width - width) // 2
            start_y = (new_height - height) // 2
            scaled_img = scaled_img[start_y:start_y + height, start_x:start_x + width]
        
        # Resize again to ensure the final size is exactly 256x256
        scaled_img = cv2.resize(scaled_img, (width, height), interpolation=cv2.INTER_LINEAR)
        return scaled_img
    

    def rotate_image(self, img, max_angle=180):
        """
        Rotate the image randomly within the specified angle range, ensuring objects don't get cut off.
        """
        # Get the dimensions of the image
        height, width, _ = self.img_shape

        # Calculate the center of the image
        center = (width // 2, height // 2)

        # Randomly choose a rotation angle
        angle = np.random.uniform(-max_angle, max_angle)

        # Calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the rotation
        border_color =  tuple(int(c) for c in img[0, 0]) 
        rotated_img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)

        return rotated_img
    
    
    def translate_image(self, img, max_translation=70):
        """
        Translate the image randomly within the specified range, ensuring objects don't get cut off.
        """
        border_color =  tuple(int(c) for c in img[0, 0]) 
        # Get the dimensions of the image
        height, width, _ = self.img_shape

        # Randomly choose translation values
        tx = np.random.randint(-max_translation, max_translation + 1)
        ty = np.random.randint(-max_translation, max_translation + 1)

        # Create a translation matrix
        M = np.float32([[1, 0, tx], [0, 1, ty]])

        # Apply the translation
        translated_img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)

        return translated_img
    
    def transform_image(self, img):
        if img.shape != self.img_shape:
            img = cv2.resize(img, (self.img_shape[0],self.img_shape[1]), interpolation=cv2.INTER_LINEAR)
        img = self.scale_image(img, max_scale=1.1)
        img = self.rotate_image(img)
        img = self.translate_image(img, max_translation=40)
        return img

    def augmentateData(self, new_samples_per_image=30, images_list = "All", sobreescribir=False):
        if images_list == "All":
            images_list = os.listdir(self.directorio_imagenes)
        for nombre_archivo in images_list:
            for i in range(new_samples_per_image):
                ruta_imagen = os.path.join(self.directorio_imagenes, nombre_archivo)
                
                ruta_salida = os.path.join(self.directorio_salida, f"{nombre_archivo[:-4]}_{i}.png")
                if os.path.exists(ruta_salida) and not sobreescribir:
                    print(f"Ya existe una imagen con el nombre {nombre_archivo}. Se omite el procesamiento.")
                else:    
                    # Cargar la imagen
                    img = cv2.imread(ruta_imagen)
                    if img is None:
                        print(f"No se pudo cargar la imagen {nombre_archivo}. Verifica la ruta.")
                    
                    # Transformar imagen
                    img = self.transform_image(img)

                    # Guardar la imagen
                    cv2.imwrite(ruta_salida, img)
                    print(f"Imagen guardada en {ruta_salida}")

#"""
if __name__ == "__main__":
    augmentator = DataAugmentator("RawTools_test", "TestSet")

    augmentator.augmentateData(new_samples_per_image=30, images_list="All", sobreescribir=False)
    
    """
    # Para nivelar la cantidad de empty images:
    # Cargar la imagen
    path = "Desarrollo/simulation/Env03/DataSets/TestSet_masks/"
    img = cv2.imread(path+"empty_white_mask.png", cv2.IMREAD_GRAYSCALE)
    # Guardar la imagen
    for i in range(1,30):
        cv2.imwrite(path+f"empty_white_mask{i}.png", img)"""