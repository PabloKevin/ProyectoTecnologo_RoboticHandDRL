from DataSet_editor import DataSet_editor
from fastapi import FastAPI
import numpy as np
import uvicorn
from pydantic import BaseModel
import os
import cv2
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk
from queue import Queue

app = FastAPI()

class ImageGenerator():
    def __init__(self, images_of_interest='all', images_dir="DataSets/RawTools/"):
        self.images_dir = images_dir
        self.image_files = [f for f in os.listdir(images_dir)] if images_of_interest == 'all' else [f for f in os.listdir(images_dir) if f in images_of_interest]
        self.image_queue = Queue()  # Queue to communicate between threads

        # Initialize tkinter window
        self.root = tk.Tk()
        self.root.title("Image Viewer")
        self.label = tk.Label(self.root)
        self.label.pack()

        # Start checking for new images
        self.root.after(100, self.check_for_new_image)

    def get_image(self, tool_name=None):
        images = [f for f in self.image_files if f.startswith(tool_name)] if tool_name is not None else self.image_files
        if not images:
            raise FileNotFoundError("No images found matching the criteria.")

        selected_image_path = os.path.join(self.images_dir, np.random.choice(images))
        #img = cv2.imread(selected_image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(selected_image_path)
        # Convert the image from BGR to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img is None:
            raise FileNotFoundError(f"Image could not be loaded from path: {selected_image_path}")

        """editor = DataSet_editor()
        img = editor.transform_image(img)

        img[img < 255 / 2] = 0  
        img[img >= 255 / 2] = 1
        img = np.expand_dims(img, axis=0)"""

        self.image_queue.put(img)  # Add the new image to the queue
        print(selected_image_path)
        return img
    
    def test_image(self, path="DataSets/Pruebas/"):
        images = [f for f in os.listdir(path)]
        if not images:
            raise FileNotFoundError("No images found matching the criteria.")

        selected_image_path = os.path.join(path, np.random.choice(images))
        img = cv2.imread(selected_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image could not be loaded from path: {selected_image_path}")

        #editor = DataSet_editor()
        #img = editor.transform_image(img)

        img[img < 255 / 2] = 0  
        img[img >= 255 / 2] = 1
        img = np.expand_dims(img, axis=0)

        self.image_queue.put(img)  # Add the new image to the queue
        print(selected_image_path)
        return img
    
    def check_for_new_image(self):
        # Check if there's a new image in the queue
        if not self.image_queue.empty():
            img_array = self.image_queue.get()
            self.update_image(img_array)

        self.root.after(100, self.check_for_new_image)  # Keep checking every 100ms

    def update_image(self, img_array):
        # Convert NumPy array to PIL Image
        #img = Image.fromarray((img_array.squeeze() * 255).astype(np.uint8))
        img = Image.fromarray(img_array)

        img = ImageTk.PhotoImage(image=img)

        # Update the tkinter label with the new image
        self.label.config(image=img)
        self.label.image = img  # Keep a reference to prevent garbage collection


# Define request structure
class ImageRequest(BaseModel):
    img_of_interest: str
    tool_name: str | None # Allow None as a valid value

imageGenerator = ImageGenerator()

@app.post("/image")
async def get_observation_img(request: ImageRequest):
    img_of_interest = request.img_of_interest
    tool_name = request.tool_name
    imageGenerator.images_of_interest = img_of_interest
    image = imageGenerator.get_image(tool_name=tool_name) 
    #image = imageGenerator.test_image()
    return {"image": image.tolist()}  # Ensure JSON serializable

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    # Run the FastAPI server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Run the tkinter mainloop in the main thread
    try:
        imageGenerator.root.mainloop()
    except KeyboardInterrupt:
        print("\nShutting down...")
