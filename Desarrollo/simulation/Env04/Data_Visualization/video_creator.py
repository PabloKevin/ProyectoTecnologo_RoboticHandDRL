import os
import cv2

def images_to_video(image_dir: str, output_path: str, fps: int = 1):
    # 1) Listar y ordenar nombres de imagen
    files = os.listdir(image_dir)

    def num_epochs(fname):
        # Extraer el número de epochs del nombre del archivo
        parts = fname.split("_")
        try:
            return int(parts[-1][:-4])
        except:
            return float('inf')
    
    files = sorted(files, key=num_epochs)

    if not files:
        raise ValueError(f"No hay imágenes en {image_dir}")

    # 2) Leer la primera imagen para sacar tamaño
    first_path = os.path.join(image_dir, files[0])
    img = cv2.imread(first_path)
    height, width = img.shape[:2]

    # 3) Definir codec y VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec para .mp4
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 4) Añadir cada imagen al video
    for fname in files:
        path = os.path.join(image_dir, fname)
        img = cv2.imread(path)
        writer.write(img)

    # 5) Liberar recursos
    writer.release()
    print(f"Video guardado en {output_path}")

if __name__ == "__main__":
    carpeta = "Desarrollo/Documentacion/actor/train_set/confusion_matrices"
    salida  = "Desarrollo/Documentacion/videos/actor_trainset_confusion_matrices.mp4"
    images_to_video(carpeta, salida, fps=1)  # 1 fps → 1s por imagen
