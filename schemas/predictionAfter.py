from pydantic import BaseModel
from typing import List,Optional


class ImageCropped(BaseModel):
    imagePerBowlData : Optional[str] = None
    remainingPercentage : float
    isRice : bool