import cv2
import numpy as np
import os

# Directorio de imágenes originales
directorio_imagenes = "Desarrollo/simulation/Env01/DataSets/RawTools/"

# Directorio para guardar las imágenes procesadas
directorio_salida = "Desarrollo/simulation/Env01/DataSets/B&W_Tools/"

# Asegurarse de que el directorio de salida existe
os.makedirs(directorio_salida, exist_ok=True)

# Iterar sobre cada archivo en el directorio
for nombre_archivo in os.listdir(directorio_imagenes):
    ruta_imagen = os.path.join(directorio_imagenes, nombre_archivo)
    
    ruta_salida = os.path.join(directorio_salida, f"bw_{nombre_archivo}")
    if os.path.exists(ruta_salida):
        print(f"Ya existe una imagen con el nombre {ruta_salida}. Se omite el procesamiento.")
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
        umbral = 10

        # Crear máscara: True donde difiere del fondo (martillo), False donde es fondo
        mascara = diferencia > umbral

        # Crear la imagen binaria (0 = negro, 255 = blanco)
        img_binaria = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        img_binaria[mascara] = 255

        # Guardar la imagen resultante
        cv2.imwrite(ruta_salida, img_binaria)

        print(f"Imagen procesada y guardada: {ruta_salida}")

    # Si se quiere visualizar (opcional):
    # cv2.imshow("Resultado", img_binaria)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
