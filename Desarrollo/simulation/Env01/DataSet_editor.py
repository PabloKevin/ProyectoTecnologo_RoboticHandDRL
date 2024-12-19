import cv2
import numpy as np
import os

class DataSet_editor:
    def __init__(self):
        # Directorio de imágenes originales
        self.directorio_imagenes = "Desarrollo/simulation/Env01/DataSets/RawTools/"

        # Directorio para guardar las imágenes procesadas
        self.directorio_salida = "Desarrollo/simulation/Env01/DataSets/B&W_Tools/"

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

                # Guardar la imagen resultante
                cv2.imwrite(ruta_salida, img_binaria)

                print(f"Imagen procesada y guardada: {ruta_salida}")

if __name__ == "__main__":
    editor = DataSet_editor()
    editor.raw2bw(sobreescribir=False, umbral=5)