from transformers import pipeline, Pipeline
from googleapiclient.discovery import build
from fastapi import FastAPI, Depends
from typing import Annotated
from torch.cuda import is_available
from pydantic import BaseModel, HttpUrl, field_validator
import re
import logging as log



def load_model() -> Pipeline: 
    
    #check if gpu is available
    if is_available():
        device = "cuda:0" # set at zero for my specific hardware.
    else:
        device = "cpu"

    pipe = pipeline("text-classification",
                    "api/model",
                    device=device,
                    max_length=512,
                    truncation=True)
    log.info("Model loaded") 
    return pipe

categories = {
    "1": "Film & Animation",
    "2": "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "18": "Short Movies",
    "19": "Travel & Events",
    "20": "Gaming",
    "21": "Videoblogging",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
    "30": "Movies",
    "31": "Anime/Animation",
    "32": "Action/Adventure",
    "33": "Classics",
    "34": "Comedy",
    "35": "Documentary",
    "36": "Drama",
    "37": "Family",
    "38": "Foreign",
    "39": "Horror",
    "40": "Sci-Fi/Fantasy",
    "41": "Thriller",
    "42": "Shorts",
    "43": "Shows",
    "44": "Trailers",
}

ModelDep = Annotated[Pipeline, Depends(load_model)]


class YoutubeUrl(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_youtube(cls, url: str) -> str:
        #pattern = r"""^(http(s)?://)? # protocol
        #            (?:(www.|m.)?youtube.com/watch\?v=|youtu.be/) # domain
        #            [a-zA-Z0-9\-_]{11} # video key
        #            .*$ # other parameters
        #            """
        pattern = r"^(http(s)?://)?(?:(www.|m.)?youtube.com/watch\?v=|youtu.be/)[a-zA-Z0-9\-_]{11}.*$"

        if re.match(pattern, url):
            return url
        else:
            raise ValueError("Provided string is not valid youtube url")
        
    @property
    def video_key(self)-> str:
        return re.search("(?<=v=)[a-zA-Z0-9\-_]{11}|(?<=youtu\.be/)[a-zA-Z0-9\-_]{11}", self.url).group()
