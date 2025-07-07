import torch
import polars as pl
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from typing import Tuple
from typing_extensions import Annotated

class Data(Dataset):
    def __init__(self, file_path :str):
        super().__init__()

        self.dataframe = pl.read_csv(file_path,
                                     separator="|")

        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        
        self.onehot = {"Positive": torch.tensor([1, 0, 0]),
                       "Neutral" : torch.tensor([0, 1, 0]),
                       "Negative": torch.tensor([0 ,0, 1])}

    def __len__(self) -> int:
        return self.dataframe.height
    
    def __getitem__(self, index :int) -> Tuple[
                                    Annotated[torch.Tensor, "data"], 
                                    Annotated[torch.Tensor, "target"]]:
        
        data, target = self.dataframe.row(index)
        
        #data = self.tokenizer.encode(data, 
        #                             return_tensors="pt",
        #                             padding=True,
        #                             truncation=True)

        data = self.tokenizer(data,
                              return_tensors="pt",
                              padding="max_length",
                              truncation=True,
                              max_length=100)
        
        target = self.onehot[target]
        print(type(target))
        return (data["input_ids"], target)