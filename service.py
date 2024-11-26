import io
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from src.utils import question_analysis

from template.request_format import RequestItem

load_dotenv()
app = FastAPI(title="matalens-api")

logger = logging.getLogger('uvicorn.error')

@app.post("/predict")
async def predict(item: RequestItem):
    return await question_analysis(item.image_path, logger)

