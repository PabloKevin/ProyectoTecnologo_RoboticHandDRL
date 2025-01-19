import cv2
import numpy as np
import os

class DataSet_editor:
    def __init__(self, img_width=256, img_height=256):
        # Directorio de imágenes originales
        self.directorio_imagenes = "Desarrollo/simulation/Env03/DataSets/RawTools/"

        # Directorio para guardar las imágenes procesadas
        self.directorio_salida = "Desarrollo/simulation/Env03/DataSets/B&W_Tools/"

        self.img_width = img_width
        self.img_height = img_height
        # Asegurarse de que el directorio de salida existe y crearlo si no existe
        os.makedirs(self.directorio_salida, exist_ok=True)

    def raw2bw(self, images_list = "All", sobreescribir=False, umbral=6):
        if images_list == "All":
            images_list = os.listdir(self.directorio_imagenes)
        for nombre_archivo in images_list:
            ruta_imagen = os.path.join(self.directorio_imagenes, nombre_archivo)
            
            ruta_salida = os.path.join(self.directorio_salida, f"bw_{nombre_archivo}")
            if os.path.exists(ruta_salida) and not sobreescribir:
                print(f"Ya existe una imagen con el nombre {nombre_archivo}. Se omite el procesamiento.")
            else:    
                # Cargar la imagen
                img = cv2.imread(ruta_imagen)
                if img is None:
                    print(f"No se pudo cargar la imagen {nombre_archivo}. Verifica la ruta.")
                
                # Redimensionar la imagen a 256x256
                img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
                
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
                # Apply morphological closing to fill small holes and connect nearby white pixels
                kernel = np.ones((3,3), np.uint8)
                img_binaria = cv2.morphologyEx(img_binaria, cv2.MORPH_CLOSE, kernel)
                
                # Iterate through pixels (excluding borders)
                for i in range(1, img_binaria.shape[0]-1):
                    for j in range(1, img_binaria.shape[1]-1):
                        # Get 8-connected neighbors
                        neighbors = img_binaria[i-1:i+2, j-1:j+2]
                        # Count white pixels in neighborhood
                        white_count = np.sum(neighbors == 255)
                        # If majority of neighbors are white, make pixel white
                        if white_count >= 5:  # Threshold of 5 out of 9 pixels
                            img_binaria[i,j] = 255

                # Convertir la imagen a escala de grises
                img_binaria = cv2.cvtColor(img_binaria, cv2.COLOR_GRAY2BGR)

                # Guardar la imagen resultanteS
                cv2.imwrite(ruta_salida, img_binaria)

                print(f"Imagen procesada y guardada: {ruta_salida}")

    def scale_image(self, img, min_scale=0.9, max_scale=1.2):
        """
        Scale the image randomly within the specified range, ensuring objects don't get cut off.
        """
        # Get the dimensions of the image
        height, width = img.shape  # Unpack the third value for color channels

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
            scaled_img = cv2.copyMakeBorder(scaled_img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
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
        height, width = img.shape

        # Calculate the center of the image
        center = (width // 2, height // 2)

        # Randomly choose a rotation angle
        angle = np.random.uniform(-max_angle, max_angle)

        # Calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the rotation
        rotated_img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return rotated_img
    
    
    def translate_image(self, img, max_translation=100):
        """
        Translate the image randomly within the specified range, ensuring objects don't get cut off.
        """
        # Convert to grayscale if the image is not already
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get the dimensions of the image
        height, width = img.shape

        # Calculate the bounding box of the white pixels
        white_pixels = np.argwhere(img == 255)

        # Check if there are any white pixels
        if white_pixels.size == 0:
            #print("No white pixels found in the image. Skipping translation.")
            return img

        min_y, min_x = white_pixels.min(axis=0)
        max_y, max_x = white_pixels.max(axis=0)

        # Calculate the maximum possible translation without cutting off the object
        max_x_translation = min(max_translation, min_x)
        max_y_translation = min(max_translation, min_y)
        max_x_translation = min(max_x_translation, width - max_x - 1)
        max_y_translation = min(max_y_translation, height - max_y - 1)

        # Randomly choose translation values
        tx = np.random.randint(-max_x_translation, max_x_translation + 1)
        ty = np.random.randint(-max_y_translation, max_y_translation + 1)

        # Create a translation matrix
        M = np.float32([[1, 0, tx], [0, 1, ty]])

        # Apply the translation
        translated_img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return translated_img
    
    def transform_image(self, img):
        img = self.scale_image(img)
        img = self.rotate_image(img)
        img = self.translate_image(img)
        return img


"""
if __name__ == "__main__":
    editor = DataSet_editor()
    #editor.raw2bw(images_list="empty.png", sobreescribir=True, umbral=10)
    
    ruta = os.path.join(editor.directorio_salida, "bw_Lapicera01.jpg")
    img = cv2.imread(ruta)
    t_img = editor.transform_image(img)
    import matplotlib.pyplot as plt 
    plt.imshow(t_img, cmap='gray')
    plt.title('Imagen Inicial')
    plt.axis('off')  # Ocultar los ejes
    plt.show()
    """