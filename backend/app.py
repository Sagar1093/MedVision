import json

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from backend.predictor import Predictor
from backend.utils import preprocess_image
from backend.schemas import (
    PredictionResponse,
    ChatRequest,
    ChatResponse
)
from backend.rag.rag_pipeline import (
    ask_question,
    stream_question
)


app = FastAPI(
    title="MedVision API",
    version="1.0.0"
)


predictor = Predictor()


@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image_tensor, rgb_image = preprocess_image(
        image_bytes
    )

    result = predictor.predict(
        image_tensor,
        rgb_image
    )

    return result


@app.post(
    "/chat",
    response_model=ChatResponse
)
async def chat(request: ChatRequest):

    result = ask_question(
        question=request.question,
        prediction=request.prediction
    )

    return result


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):

    def generate():

        for item in stream_question(
            question=request.question,
            prediction=request.prediction
        ):
            yield json.dumps(item) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )