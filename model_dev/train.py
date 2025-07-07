import torch 
from torch.utils.data import DataLoader
from dataset import Data


if __name__ == "__main__":
    data = Data("./dataset/pure_comments.csv")
    comment, target = data[56]
    dataloader = DataLoader(data, batch_size=32)

    for com,tar in dataloader:
        comment = com
        target = tar
        break
    
    print(comment)
    print(target)