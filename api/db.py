from sqlmodel import Field, Session, SQLModel, create_engine
from typing import Annotated
import os

DB_URL = "sqlite:///database.db"
CONNECTION_ARGS = {"check_same_thread": False}

class Video(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    videoName: str
    channelName: str
    videoKey: str
    viewCount: int
    likeCount: int
    commentCount: int

engine = create_engine(DB_URL, connect_args=CONNECTION_ARGS)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session