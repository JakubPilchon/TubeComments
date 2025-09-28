from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ..db import SessionDep, Video, Comment, Sentiment
from sqlmodel import select, func, desc


info = APIRouter(
    prefix="/info",
    tags=["info"]
)

@info.get("/video/{id}")
def get_video_info(id : int, session: SessionDep):
    """
    Return an information about selected Video

    Parameters:
    - **id** - id of video in database
    """
    video = session.get(Video, id)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return video

@info.get("/mostLikedComments/{id}")
def get_most_liked_comments(
    id : int,
    session: SessionDep,
    limit: int = 5):
    """
    Returns most liked comments from selected video across all sentiments

    Parameters:
    - **id** - id of video in database
    - **limit** - max number of comment returned
    """
    query = (
        select(Comment.commentText, Comment.likeCount, Comment.sentiment)
        .where(Comment.videoId == id)
        .order_by(desc(Comment.likeCount))
        .limit(limit)
        )
    
    result = session.exec(query).mappings().all()

    return result
