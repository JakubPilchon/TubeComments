from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ..db import get_session, SessionDep, Video, Comment
from ..utils import ModelDep, categories
from sqlmodel import select, delete
from googleapiclient.discovery import build
import regex
import os

API_KEY = os.getenv("YOUTUBE_API_KEY")
assert API_KEY is not None

index = APIRouter(
    prefix="/menu",
    tags=["Menu"]
)

@index.get("/getVideos")
def get_videos(session: SessionDep):
    """
    Retrieve basic information about videos in the database.
    Returns title of the video, authors channel name and id of the video in database.

    """
    query = select(Video.videoName, Video.id, Video.channelName)
    videos = session.exec(query).mappings().all()
    return videos

@index.post("/getVideoInfo", status_code=204)
def post_film(video_link: str,
              session: SessionDep,
              model: ModelDep):
    """
    Downloads data about selected video from youtube api, scrapes comments from video. 
    Then uploads daat to database.

    Parameters:
    - **video_link** - link to youtube video
    """

    video_key = regex.search("(?<=v=).{11}|(?<=youtu\.be/).{11}", video_link).group()
    print(video_key)
    if video_key is None:
        raise HTTPException(status_code=400,
                            detail="Video Key not found in provided link")

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
    

@index.delete("/deleteVideo", status_code=204)
def delete_video(id: int, session: SessionDep):
    """Deletes both video and comments info from the database.
    
    Parameters:
    - **id** - id of video in database
    """

    query = delete(Video).where(Video.id == id)
    session.exec(query)

    query = delete(Comment).where(Comment.videoId == id)
    session.exec(query)

    session.commit()