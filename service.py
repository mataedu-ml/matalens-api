import io
from fastapi import FastAPI, Request
from src.utils import question_analysis

from pydantic import BaseModel

class Item(BaseModel):
    image_path: str


app = FastAPI()

@app.post("/predict")
async def predict(item: Item):
    return question_analysis(item.image_path)

