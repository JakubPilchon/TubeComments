import mlflow.pytorch
import torch
import os
import mlflow
import lightning as L
import polars as pl
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from dataset import Data
from model import Model


EXPERIMENT_NAME = "Lstm_comments"
EXPERIMENT_DESCRIPTION = "Predicting sentiment of youtube comments."
BATCH_SIZE = 512
SEQUENCE_LENGTH = 40
cli = mlflow.client.MlflowClient()

#disable tokenizer parallelism, to avoid deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":

    mlflow.set_experiment("Lstm_comments")


    with mlflow.start_run() as run:
        data = Data("./dataset/youtube-comments-sentiment.csv",
                    seq_length = SEQUENCE_LENGTH,
                    lang_path = "./dataset/language.csv")
        
        trainingset, testset = random_split(data, [0.8, 0.2])

        train_loader = DataLoader(trainingset, BATCH_SIZE, num_workers=11)
        test_loader = DataLoader(testset, BATCH_SIZE, num_workers=11)

        hyperparameters = {
            "hidden_dim_1" : 512,
            "hidden_dim_2" : 256,
            "embedding_dim": 512,
            "dropout_rate" : 0.3,
            "learning_rate": 4e-3
        }

        mlflow.log_params(hyperparameters)
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("batch_size", BATCH_SIZE)

        model = Model(**hyperparameters)

        logger = MLFlowLogger(EXPERIMENT_NAME,
                              run_id=run.info.run_id,
                              save_dir="mlruns",
                              log_model=False
                              )
        
        ea_call = EarlyStopping("val_loss", patience=1)
        mc_call = ModelCheckpoint(
            dirpath="model_dev/saved_models",
            filename=f'{run.info.run_name}' + '-{epoch}-{val_accuracy:.2f}'
        )

        T = L.Trainer(logger=logger,
                      callbacks=[ea_call],
                      max_epochs=10,
                      devices=[0]
                      )

        T.fit(model, train_loader, test_loader)

        mlflow.pytorch.log_model(model, name="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=f"LSTM-{run.info.run_name}")

        predictions = T.predict(model, test_loader)
        predictions = list(map(testset.dataset.rev_onehot.get,
                               torch.cat(predictions, dim=0).reshape(-1).tolist()))
        








