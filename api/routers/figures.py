from fastapi import APIRouter, HTTPException
from api.db import SessionDep, Comment
from sqlmodel import select, func, and_
from datetime import datetime

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

@figures.get("/getPublishDates/{id}")
def get_publishing_dates(
    id: int,
    lower_date: datetime,
    higher_date: datetime,
    session: SessionDep):
    """Returns a lists of comment publishing dates with corresponding sentiment
    
    Parameters:
    - **id** - Id of video in database
    - **lower_date** - dates up to this date will be filtered out
    - **higher_date** - dates following this date will be filtered out
    """

    query = (
        select(Comment.sentiment, Comment.publishingDate)
        .where(
            and_(Comment.videoId == id,
               Comment.publishingDate >= lower_date,
               Comment.publishingDate <= higher_date)
        )
    )

    result = {
        "date" : [],
        "sentiment" : []
    }

    for sentiment, date in session.exec(query):
        result["date"].append(date)
        result["sentiment"].append(sentiment)

    return result
    
    


