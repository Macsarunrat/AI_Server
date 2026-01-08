from pydantic import BaseModel
from typing import List, Optional

class ImagePredictionBefore(BaseModel):
    imageData : Optional[str] = None
    predictBowlMenu : str
    confidence : float
    

class PredictionResponse(BaseModel):
    predictions : List[ImagePredictionBefore]




