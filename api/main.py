from fastapi import FastAPI, Depends, HTTPException
from googleapiclient.discovery import build
from transformers import pipeline
from funcs import load_model, init_youtube_api
from db import init_db, get_session, Video
from sqlmodel import Session
from typing import Annotated
import logging as log
import os

SessionDep = Annotated[Session, Depends(get_session)]

API_KEY = os.getenv("YOUTUBE_API_KEY")

def lifespan(app: FastAPI):
    app.state.model = load_model()
    init_db()
    yield


app = FastAPI(lifespan=lifespan)

@app.get("/video/{id}")
def main(id : int, session: SessionDep):
    video = session.get(Video, id)
    if not video:
        raise HTTPException(status_code=404, detail="Hero not found")
    return video

@app.post("/getVideoInfo")
def post_film(video_id: str, session: SessionDep):

    youtube = build(serviceName="youtube",
                        version="v3",
                        developerKey=API_KEY)
    
    request = youtube.videos().list(
    part="snippet,contentDetails,statistics",
    id=video_id
    )
    response = request.execute()

    session.add(
        Video(videoName    = response["items"][0]["snippet"]["title"],
              channelName  = response["items"][0]["snippet"]["channelTitle"],
              videoKey     = video_id,
              viewCount    = response["items"][0]["statistics"]["viewCount"],
              likeCount    = response["items"][0]["statistics"]["likeCount"],
              commentCount = response["items"][0]["statistics"]["commentCount"]
              ))
    
    session.commit()
    
    return response
    
