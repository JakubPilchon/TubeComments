from fastapi import FastAPI
from googleapiclient.discovery import build
from transformers import pipeline
from funcs import load_model, init_youtube_api
import logging as log


app = FastAPI(lifespan=load_model)

@app.get("/")
def main(text : str):
    result = app.state.model(text)
    return result

    
