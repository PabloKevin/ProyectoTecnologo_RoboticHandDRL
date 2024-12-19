import cv2
import numpy as np

# Ruta de la imagen original
ruta_imagen = "Desarrollo/simulation/Env01/DataSets/RawTools/Martillo01.jpg"  # Ajusta el nombre y la ruta según sea necesario

# Cargar la imagen
img = cv2.imread(ruta_imagen)
if img is None:
    print("No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# Redimensionar la imagen a 64x64
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
cv2.imwrite("Desarrollo/simulation/Env01/DataSets/B&W_Tools/empty.png", img_binaria)

# Si se quiere visualizar (opcional):
# cv2.imshow("Resultado", img_binaria)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
