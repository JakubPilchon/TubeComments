from sqlmodel import (Field, Session, SQLModel, create_engine,
                      Column, Enum)
from typing import Annotated
from fastapi import Depends
from datetime import datetime
from pydantic import computed_field
import os
import enum

pwd = os.getcwd()
DB_URL = f"sqlite:///{pwd}/database.db"
CONNECTION_ARGS = {"check_same_thread": False}

class Sentiment(str, enum.Enum):
    Positive = "Positive"
    Neutral = "Neutral"
    Negative = "Negative"
 

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

    @computed_field
    @property
    def engagementRate(self) -> float:
        return ((self.likeCount + self.commentCount) / self.viewCount
                if self.viewCount != 0 else .0)

class Comment(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    videoId: int = Field(default=None, foreign_key="video.id", index=True)
    commentText: str
    sentiment: Sentiment = Field(sa_column=Column(Enum(Sentiment)))
    likeCount: int
    publishingDate: datetime

engine = create_engine(DB_URL, connect_args=CONNECTION_ARGS, echo=True)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]