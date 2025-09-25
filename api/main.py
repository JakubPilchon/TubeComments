from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from googleapiclient.discovery import build
from utils import load_model, categories, ModelDep
from db import init_db, get_session, Video, Comment
from sqlmodel import Session, select
from typing import Annotated
import logging
import os

SessionDep = Annotated[Session, Depends(get_session)]

API_KEY = os.getenv("YOUTUBE_API_KEY")
assert API_KEY is not None

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

def lifespan(app: FastAPI):
    app.state.model = load_model()
    init_db()
    yield


app = FastAPI(lifespan=lifespan)

@app.get("/commentList/{id}")
def get_comments(id: int, session: SessionDep):
    query = (select(Comment)
             .where(Comment.videoId == id))
    result = session.exec(query)
    comments = []
    for com in result:
        comments.append(com)

    if not comments:
        raise HTTPException(status_code=404, 
                            detail="Comment for the video were not found")

    return comments

@app.get("/video/{id}")
def main(id : int, session: SessionDep):
    video = session.get(Video, id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

@app.post("/getVideoInfo", status_code=201)
def post_film(video_key: str,
              session: SessionDep,
              model: ModelDep):

    youtube = build(serviceName="youtube",
                        version="v3",
                        developerKey=API_KEY)
    
    request = youtube.videos().list(
    part="snippet,contentDetails,statistics",
    id=video_key
    )
    response = request.execute()

    if response["pageInfo"]["totalResults"] == 0:
        raise HTTPException(status_code=400,
                             detail="Invalid video_key")

    video = Video(videoName    = response["items"][0]["snippet"]["title"],
                  channelName  = response["items"][0]["snippet"]["channelTitle"],
                  videoKey     = video_key,
                  category     = categories.get(response["items"][0]["snippet"]["categoryId"], "No category"),
                  viewCount    = response["items"][0]["statistics"]["viewCount"],
                  likeCount    = response["items"][0]["statistics"]["likeCount"],
                  commentCount = response["items"][0]["statistics"]["commentCount"])
    
    session.add(video)
    session.commit()
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId = video_key,
        textFormat="plainText"
    )

    while request:
        try:
            response = request.execute()
            for item in response["items"]:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                text = snippet["textDisplay"]

                sentiment = model(text)[0]["label"]
                comm = Comment(
                    commentText=text,
                    sentiment=sentiment,
                    likeCount=snippet["likeCount"],
                    videoId=video.id
                )

                session.add(comm)
                
            request = youtube.comments().list_next(request, response)
        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=str(e))

    session.commit()
    
    return {"result": "ok"}
    
