import os
import cv2

def images_to_video(image_dir: str, output_path: str, fps: int = 30, duration: list = [1, 0.5, 0.2], threshold_files: list = [16, 25]):
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
    for i, fname in enumerate(files):
        path = os.path.join(image_dir, fname)
        img = cv2.imread(path)

        for j, th in enumerate(threshold_files):
            if i < th:
                time = duration[j]
                break
            elif i >= threshold_files[-1]:
                time = duration[-1]

        n_frames = int(time * fps)
        for _ in range(n_frames):
            writer.write(img)


    # 5) Liberar recursos
    writer.release()
    print(f"Video guardado en {output_path}")

if __name__ == "__main__":
    # nn architecture videos
    """
    for version in ["v2_trainset", "v2_fullset"]:
        for nn in ["actor", "critic_1", "critic_2", "target_actor", "target_critic_1", "target_critic_2"]:
            carpeta = f"Desarrollo/Documentacion/nn_arch/{version}/{nn}/"
            salida  = f"Desarrollo/Documentacion/videos/nn_arch_trinset/{nn}_{version[3:]}_nn_arch_.mp4"
            images_to_video(carpeta, salida, fps=5, duration=[0.8, 0.4, 0.2], threshold_files=[16,25])  """
    
    # td3 confusion matrices videos
    """for version in ["train_set", "full_set"]:
        carpeta = f"Desarrollo/Documentacion/actor/{version}/confusion_matrices/"
        salida  = f"Desarrollo/Documentacion/videos/actor_{version}_confusion_matrices.mp4"
        images_to_video(carpeta, salida, fps=5, duration=[0.8, 0.4, 0.2], threshold_files=[16,25])"""

    # observer confusion matrices video
    carpeta = f"Desarrollo/Documentacion/observer/confusion_matrices/"
    salida  = f"Desarrollo/Documentacion/videos/observer_confusion_matrices.mp4"
    images_to_video(carpeta, salida, fps=5, duration=[0.8, 0.4], threshold_files=[16])
