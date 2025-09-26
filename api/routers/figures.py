from fastapi import APIRouter, HTTPException
from api.db import SessionDep, Comment
from sqlmodel import select, func

figures = APIRouter(
    prefix="/figures",
    tags=["Figures"]
)

@figures.get("/getSentiments/{id}")
def get_sentiments(
                   id: int,
                   session: SessionDep):
    """Returns the sentiments of comments of selected video.
    
    Parameters:
    - **id** - Id of video in database
    """

    query = (
        select(Comment.sentiment,
               func.count(Comment.id))
        .where(Comment.videoId == id)
        .group_by(Comment.sentiment)
             )
    
    response = session.exec(query).all()

    if not response:
        raise HTTPException(status_code=404,
                            detail="Comments for the video were not found")

    return {key:value for key, value in response}

@figures.get("/getLikesInfo/{id}")
def get_likes_info(
                   id: int,
                   session: SessionDep):
    """Return the number of likes given to comments for each sentiment.
    
    Parameters:
    - **id** - Id of video in database
    """

    query = (
        select(Comment.sentiment,
               func.sum(Comment.likeCount))
        .where(Comment.videoId == id)
        .group_by(Comment.sentiment)
    )

    response = session.exec(query).all()

    if not response:
        raise HTTPException(status_code=404,
                            detail="Comments for the video were not found")

    return {key:value for key, value in response}

@figures.get("/getCommentLenghts/{id}")
def get_video_lenghts(
                   id: int,
                   session: SessionDep):
    """
    Return a lists of comment character lenghts with corresponding sentiment

    Parameters:
    - **id** - Id of video in database
    """
    
    query = (
        select(Comment.sentiment,
               func.char_length(Comment.commentText))
        .where(Comment.videoId == id)
    )

    result = {
        "commentLength" : [],
        "sentiment" : []
    }
    response = session.exec(query).all()

    for sentiment, length in response:
        result["commentLength"].append(length)
        result["sentiment"].append(sentiment)

    return result


    
    


