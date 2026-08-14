from fastapi import FastAPI,UploadFile,File
from backend.predictor import Predictor
from backend.utils import preprocess_image
from backend.schemas import PredictionResponse,ChatRequest,ChatResponse
from backend.rag.rag_pipeline import ask_question

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

    image_tensor,rgb_image = preprocess_image(image_bytes)

    result = predictor.predict(image_tensor,rgb_image)

    return result

@app.post("/chat",response_model=ChatResponse)
async def chat(request:ChatRequest):
    result = ask_question(
        question=request.question,
        prediction=request.prediction
    )

    return result
    


