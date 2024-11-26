import io
import os
import requests
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from src.utils import question_analysis

from template.request_format import RequestItem

load_dotenv()
app = FastAPI(title="matalens-api")

logger = logging.getLogger('uvicorn.error')

@app.post("/predict")
async def predict(item: RequestItem):
    return question_analysis(item.image_path, logger)

