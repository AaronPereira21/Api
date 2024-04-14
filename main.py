# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model('my_model.h5')
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

class ImageData(BaseModel):
    image_data: list

@app.post('/predict/')
async def predict(image_data: ImageData):
    try:
        image_data_np = np.array(image_data.image_data)
        # Ensure that the input shape matches the model's input shape
        if image_data_np.shape != (32, 32, 3):
            raise HTTPException(status_code=400, detail="Invalid input shape. Expected (32, 32, 3)")
        
        # Perform prediction
        prediction = model.predict(np.expand_dims(image_data_np, axis=0))
        predicted_class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        predicted_class = class_names[predicted_class_index]
        
        return {'predicted_class': predicted_class, 'confidence': confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
