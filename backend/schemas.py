from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction:str
    confidence:float
    probabilities:dict[str,float]
    heatmap:str