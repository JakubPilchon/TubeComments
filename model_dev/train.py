import torch 
import lightning as L
from torch.utils.data import DataLoader, random_split
from torch.nn import Embedding, LSTM, Dropout, Linear, Softmax
from dataset import Data
from model import Model


if __name__ == "__main__":
    data = Data("./dataset/pure_comments.csv")
    trainingset, testset = random_split(data, [0.8, 0.2])

    train_loader = DataLoader(trainingset, 256, num_workers=11)
    test_loader = DataLoader(testset, 256)

    hyperparameters = {
        "hidden_dim"   : 256,
        "embedding_dim": 128,
        "dropout_rate" : 0.2
    }

    model = Model(**hyperparameters)

    T = L.Trainer()
    T.fit(model, train_loader, test_loader)



