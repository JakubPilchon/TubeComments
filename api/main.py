from fastapi import FastAPI, Depends, HTTPException
from api.utils import load_model
from api.db import init_db, get_session, Video, Comment, SessionDep
from sqlmodel import Session, select
from api.routers import index, figures
import logging

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

def lifespan(app: FastAPI):
    app.state.model = load_model()
    init_db()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(index)
app.include_router(figures)

@app.get("/")
def get_status():
    return {"result": "app is running"}

@app.get("/commentList/{id}", deprecated=True)
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

@app.get("/video/{id}", deprecated=True)
def main(id : int, session: SessionDep):
    video = session.get(Video, id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


    
