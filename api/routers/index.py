from fastapi import APIRouter, HTTPException
from ..db import get_session, SessionDep, Video, Comment
from ..utils import ModelDep, categories
from sqlmodel import select
from googleapiclient.discovery import build
import os

API_KEY = os.getenv("YOUTUBE_API_KEY")
assert API_KEY is not None

index = APIRouter(
    prefix="/index",
    tags=["index"]
)

@index.get("/getVideos")
def get_videos(session: SessionDep):
    query = select(Video.videoName, Video.id, Video.channelName)
    videos = session.exec(query).mappings().all()
    return videos

@index.post("/getVideoInfo", status_code=201)
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