from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction:str
    confidence:float
    probabilities:dict[str,float]
    heatmap:str

class ChatRequest(BaseModel):
    question:str
    prediction:str|None = None

class Source(BaseModel):
    id:str
    source:str
    page:int

class ChatResponse(BaseModel):
    answer:str
    sources:list[Source]