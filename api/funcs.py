from transformers import pipeline, Pipeline
from googleapiclient.discovery import build
from fastapi import FastAPI
import logging as log

def load_model() -> Pipeline: 
    pipe = pipeline("text-classification", "api/model", device=0)
    log.info("Model loaded") 
    return pipe

def init_youtube_api(api_key: str):
    youtube = build(serviceName="youtube",
                        version="v3",
                        developerKey=api_key)
    log.info("Youtube api loaded")
    yield