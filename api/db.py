from sqlmodel import Field, Session, SQLModel, create_engine
from typing import Annotated
from fastapi import Depends
from datetime import datetime
import os

pwd = os.getcwd()
DB_URL = f"sqlite:///{pwd}/database.db"
CONNECTION_ARGS = {"check_same_thread": False}

class Video(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    videoName: str
    channelName: str
    videoKey: str
    category: str | None
    viewCount: int
    likeCount: int
    commentCount: int
    publishingDate: datetime

class Comment(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    videoId: int = Field(default=None, foreign_key="video.id", index=True)
    commentText: str
    sentiment: str | None
    likeCount: int
    publishingDate: datetime

engine = create_engine(DB_URL, connect_args=CONNECTION_ARGS, echo=True)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]