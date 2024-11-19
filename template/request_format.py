from typing import List
from pydantic import BaseModel

class RequestItem(BaseModel):
    image_path: List[str]
