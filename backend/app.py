from fastapi import FastAPI,UploadFile,File
from backend.predictor import Predictor
from backend.utils import preprocess_image
from backend.schemas import PredictionResponse

app = FastAPI(
    title="MedVision API",
    version="1.0.0"
)

predictor = Predictor()

@app.get("/health")
def health():
    return {
        "Status":"Healthy"
    }

@app.post("/predict",
          response_model=PredictionResponse
          )
async def predict(file:UploadFile = File(...)):
    image_bytes = await file.read()

    image = preprocess_image(image_bytes)

    result = predictor.predict(image)

    return result
    


