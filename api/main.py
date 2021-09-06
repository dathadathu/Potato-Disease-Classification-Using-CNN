from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
import numpy as np
from PIL import Image

import tensorflow as tf

MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ["Early Blight", "Late_Blight", "Healthy"]



app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello, I'm alive"


def read_file_as_image(data) ->  np.ndarray:
    image = np.array(Image.open(BytesIO(data)))

    return image 


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch =  np.expand_dims(image,0)

    predictions = MODEL.predict(image)

    predicted_class =  CLASS_NAMES[np.argmax(predictions[0])]

    confidence = np.max(predictions[0])

    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
