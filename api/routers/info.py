from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ..db import SessionDep, Video, Comment, Sentiment
from ..utils import stopwords
from sqlmodel import select, func, desc, and_
from collections import Counter
import string

info = APIRouter(
    prefix="/info",
    tags=["Info"]
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

@info.get("/mostCommonWords")
def get_most_common_word(
    id: int,
    sentiment: Sentiment,
    session: SessionDep,
    limit: int = 10):
    """
    Counters words in every comment of the video with defined sentiment.

    Parameters:
    - **id** - id of video in database
    - **sentiment** - sentiment of which comments wil be used in the counting process
    - **limit** - max number of comment returned
    """

    word_counter = Counter()
    query = (
        select(Comment.commentText)
        .where(and_(Comment.videoId == id, Comment.sentiment == sentiment))
    )

    result = session.exec(query)
    translator = str.maketrans('', '', string.punctuation + '1234567890')

    for text in result:
        word_counter.update(text
                             .translate(translator)
                             .lower()
                             .split())
    if result:
        for stopword in stopwords:
            word_counter[stopword] = 0

    most_common = {word: count for word, count in word_counter.most_common(limit)}

    return most_common