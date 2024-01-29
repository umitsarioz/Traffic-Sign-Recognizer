import uvicorn
import tensorflow
import numpy as np
import os.path
from keras.models import load_model
from fastapi import FastAPI
from pydantic import BaseModel
try:
    from generate_traffic_sign import predict
    model_fp = os.path.join('models', 'ai_model.h5')
except:
    from service.generate_traffic_sign import predict
    model_fp = os.path.join('service','models', 'ai_model.h5')

model = load_model(filepath=model_fp)
app = FastAPI()

class ImageModel(BaseModel):
    img_array: list = None


@app.get('/status')
def status():
    return {"message": "ChestXAI API is running."}


@app.post('/predict')
def predict_caption(inputs: ImageModel):
    print("trigger post request..")
    img = np.asarray(inputs.img_array)
    print("from list to np array:",img.shape,"\tType:",type(img))
    y_pred = predict(img, model)
    return {"message": y_pred}
