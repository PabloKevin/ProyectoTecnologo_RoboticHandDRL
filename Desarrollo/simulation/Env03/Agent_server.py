from Full_Agent import Full_Agent_Pipe
from fastapi import FastAPI
import numpy as np
import uvicorn
from pydantic import BaseModel

app = FastAPI()
# "Desarrollo/simulation/Env03/tmp/" # para usar en vsc
# "tmp/" # para usar en linux terminal
Agent = Full_Agent_Pipe(checkpoint_dir="models_params_weights/")

# Define request structure
class ObservationRequest(BaseModel):
    observation: list

@app.post("/predict")
async def get_action(request: ObservationRequest):
    observation = np.array(request.observation, dtype=np.uint8)
    print("Shape of incoming observation:", observation.shape)
    action = Agent.pipe(input_img=observation, render=False)  # Predict action
    return {"action": action}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Listen on all IPs, port 8000
