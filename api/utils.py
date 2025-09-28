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

ModelDep = Annotated[Pipeline, Depends(load_model)]

class YoutubeUrl(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_youtube(cls, url: str) -> str:

        pattern = r"^(http(s)?://)?(?:(www.|m.)?youtube.com/watch\?v=|youtu.be/)[a-zA-Z0-9\-_]{11}.*$"

        if re.match(pattern, url):
            return url
        else:
            raise ValueError("Provided string is not valid youtube url")
        
    @property
    def video_key(self)-> str:
        return re.search("(?<=v=)[a-zA-Z0-9\-_]{11}|(?<=youtu\.be/)[a-zA-Z0-9\-_]{11}", self.url).group()

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

stopwords =  { # chatgpt generated :-)
    "a","about","above","after","again","against","all","am","an","and","any","are",
    "arent","as","at","be","because","been","before","being","below","between","both",
    "but","by","cant","cannot","could","couldnt","did","didnt","do","does","doesnt",
    "doing","dont","down","during","each","few","for","from","further","had","hadnt",
    "has","hasnt","have","havent","having","he","hed","hell","hes","her","here",
    "heres","hers","herself","him","himself","his","how","hows","i","id","ill","im",
    "ive","if","in","into","is","isnt","it","its","itself","lets","me","more",
    "most","mustnt","my","myself","no","nor","not","of","off","on","once","only","or",
    "other","ought","our","ours","ourselves","out","over","own","same","shant","she",
    "shed","shell","shes","should","shouldnt","so","some","such","than","that",
    "thats","the","their","theirs","them","themselves","then","there","theres","these",
    "they","theyd","theyll","theyre","theyve","this","those","through","to","too",
    "under","until","up","very","was","wasnt","we","wed","well","were","weve",
    "were","werent","what","whats","when","whens","where","wheres","which","while",
    "who","whos","whom","why","whys","with","wont","would","wouldnt","you","youd",
    "youll","youre","youve","your","yours","yourself","yourselves",
    "also","just","like","one","two","three","us","get","got","may","might","much",
    "many","still","yet","ever","never","say","says","said","see","seen","go","goes",
    "went","come","comes","came","make","makes","made","know","known","thing","things",
    "take","taken","put","puts","want","wanted","use","used","using","back","even",
    "well","every","everyone","everything","because","however","although","though",
    "perhaps","rather","quite","often","always","sometimes","already","too",
    "very","else","such","within","without","across","around","behind","beyond","toward"
}