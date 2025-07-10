import lightning as L
import torch
from torch.nn import Embedding, LSTM, Dropout, Linear, Softmax


class Model(L.LightningModule):
    def __init__(self, 
                 hidden_dim : int,
                 embedding_dim: int,
                 dropout_rate: float):
        super().__init__()

        self.embedding = Embedding(num_embeddings= 50_265, # number of embedding for tokenizer used
                                   embedding_dim = embedding_dim)
        
        self.lstm = LSTM(input_size=  embedding_dim,
                         hidden_size= hidden_dim)
        
        self.dropout = Dropout(dropout_rate)

        self.lin = Linear(hidden_dim, 3)


    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x, _ = self.lstm(x) 

        x = x[:, -1, :]   
        x = self.dropout(x)
        x = self.lin(x)
        return x
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        predicted = self(data)

        loss = torch.nn.functional.cross_entropy(predicted, target.argmax(1))
        accuracy = (target.argmax(1) == predicted.argmax(1)).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        predicted = self(data)

        loss = torch.nn.functional.cross_entropy(predicted, target.argmax(1))
        accuracy = (target.argmax(1) == predicted.argmax(1)).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return {
            "optimizer" : optimizer
        }
    
    def train_dataloader(self):
        return super().train_dataloader()