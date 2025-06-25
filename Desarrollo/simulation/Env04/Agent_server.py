from Full_Agent import Full_Agent_Pipe
from fastapi import FastAPI
import numpy as np
import uvicorn
from pydantic import BaseModel
import cv2
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk
from queue import Queue

app = FastAPI()
# "Desarrollo/simulation/Env04/tmp/" # para usar en vsc
# "tmp/" # para usar en linux terminal
Agent = Full_Agent_Pipe(checkpoint_dir="models_params_weights/")

class MaskImageTK():
    def __init__(self):
        self.image_queue = Queue()  # Queue to communicate between threads

        # Initialize tkinter window
        self.root = tk.Tk()
        self.root.title("Observer Input")
        self.label = tk.Label(self.root)
        self.label.pack()

        # Start checking for new images
        self.root.after(100, self.check_for_new_image)
    
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
class ObservationRequest(BaseModel):
    observation: list

mask_image_tk = MaskImageTK()

@app.post("/predict")
async def get_action(request: ObservationRequest):
    observation = np.array(request.observation, dtype=np.uint8)
    print("Shape of incoming observation:", observation.shape)
    action, bw_mask = Agent.pipe(input_img=observation, render=False)  # Predict action
    bw_mask = bw_mask.squeeze(0)
    bw_mask = bw_mask*255
    mask_image_tk.image_queue.put(bw_mask)  # Add the new image to the queue
    return {"action": action}

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Listen on all IPs, port 8000

if __name__ == "__main__":
    # Run the FastAPI server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Run the tkinter mainloop in the main thread
    try:
        mask_image_tk.root.mainloop()
    except KeyboardInterrupt:
        print("\nShutting down...")
