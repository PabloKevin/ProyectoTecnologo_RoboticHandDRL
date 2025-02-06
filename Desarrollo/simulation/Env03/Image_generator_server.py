from DataSet_editor import DataSet_editor
from fastapi import FastAPI
import numpy as np
import uvicorn
from pydantic import BaseModel
import os
import cv2
import matplotlib.pyplot as plt
import threading

app = FastAPI()

class ImageGenerator():
    def __init__(self, images_of_interest='all', images_dir="DataSets/B&W_Tools/"):
        self.images_dir = images_dir
        
        if images_of_interest == "all":
            # List all image files in the directory
            self.image_files = [f for f in os.listdir(images_dir)]
        else:
            self.image_files = [f for f in os.listdir(images_dir) if f in images_of_interest]
        self.new_image = False
         
    def get_image(self, tool_name=None):
        if tool_name is not None:
            images = [f for f in self.image_files if f.startswith("tool_name")]
        else:
            images = self.image_files
        # Select an random image
        num_images = len(images)
        random_index = np.random.randint(0, num_images)
        selected_image_path = os.path.join(self.images_dir, images[random_index])
        
        # Load the image
        img = cv2.imread(selected_image_path, cv2.IMREAD_GRAYSCALE)
        # Transform the image so there is a "different" image for each episode
        editor = DataSet_editor()
        img = editor.transform_image(img)

        # Convert pixels values from 255 to 1
        img[img < 255/2] = 0  
        img[img >=  255/2] = 1
        #file = "Desarrollo/simulation/Env01/img.txt"
        #np.savetxt(file, img, fmt="%d", delimiter=" ") 
        
        # Add a channel dimension to the image
        img = np.expand_dims(img, axis=0)
        self.new_image = True
        self.last_image = img
        return img
    
    def render(self, render_timeout=None):
        while True:
            if self.new_image is True:
                self.new_image = False
                while self.new_image == False:
                    plt.imshow(self.last_image.squeeze(), cmap='gray')
                    plt.title('Input Image')
                    plt.axis('off') 
                    if render_timeout is not None:
                        plt.show(block=False)
                        plt.pause(render_timeout)  # Show plot for x seconds
                        plt.close()  # Close the plot window
                    else:
                        plt.show()
                plt.close()


# Define request structure
class ImageRequest(BaseModel):
    img_of_interest: str
    tool_name: str | None  # Allow None as a valid value

imageGenerator = ImageGenerator()

@app.post("/image")
async def get_observation_img(request: ImageRequest):
    img_of_interest = request.img_of_interest
    tool_name = request.tool_name
    imageGenerator.images_of_interest = img_of_interest
    image = imageGenerator.get_image(tool_name=tool_name)  # Predict action
    return {"image": image.tolist()}  # Ensure JSON serializable

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Listen on all IPs, port 8000

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server)
    plot_thread = threading.Thread(target=imageGenerator.render)


    server_thread.start()
    plot_thread.start()

    
    server_thread.join()
    plot_thread.join()
