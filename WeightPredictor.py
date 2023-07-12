from pydantic import BaseModel

class WeightPredictor(BaseModel):
    Height:float
    Gender:str