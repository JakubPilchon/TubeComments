import torch
import polars as pl
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from typing import Tuple
from typing_extensions import Annotated

class Data(Dataset):
    def __init__(self,
                  file_path :str,
                  seq_length: int,
                  lang_path: str | None = None
                ):
        
        super().__init__()
        self.seq_length = seq_length

        self.dataframe = pl.read_csv(file_path)
        
        if lang_path is not None:
            language = pl.read_csv(lang_path).to_series()
            self.dataframe = self.dataframe.insert_column(index = -1, 
                                                          column=language)
            
            self.dataframe  = self.dataframe.filter(pl.col("Language") == "en")

        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        
        self.onehot = {
            "Positive": torch.tensor(0),
            "Neutral" : torch.tensor(1),
            "Negative": torch.tensor(2)
        }

        self.rev_onehot = {
            0: "Positive",
            1: "Neutral",
            2: "Negative"
        }

    def __len__(self) -> int:
        return self.dataframe.height
    
    def tokenize(self,
                 text: str
                 ) -> torch.Tensor:
        
        text = self.tokenizer(text, 
                              return_tensors="pt",
                              padding="max_length",
                              truncation=True,
                              max_length=self.seq_length)
        
        return text["input_ids"][0]
    
    def __getitem__(self, index :int) -> Tuple[
                                    Annotated[torch.Tensor, "data"], 
                                    Annotated[torch.Tensor, "target"]]:
        
        data = self.dataframe.row(index, named=True)
        
        text = self.tokenize(data["CommentText"])
        
        target = self.onehot[data["Sentiment"]]
        
        return (text, target)