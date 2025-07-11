import torch 
import lightning as L
import mlflow 
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from torch.utils.data import DataLoader, random_split
from dataset import Data
from model import Model
import os

EXPERIMENT_NAME = "youtube_comments"
EXPERIMENT_DESCRIPTION = "Predicting sentiment of youtube comments."
BATCH_SIZE = 256
SEQUENCE_LENGTH = 150
cli = mlflow.client.MlflowClient()

#disable tokenizer parallelism, to avoid deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":

    experiment = cli.get_experiment_by_name(EXPERIMENT_NAME)
    print(experiment)

    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:     
        experiment = cli.create_experiment(EXPERIMENT_NAME,
                                           tags= {
                                               "description": EXPERIMENT_DESCRIPTION,
                                               "library": "pytorch",
                                               "task": "nlp"
                                           })
        experiment_id = experiment.experiment_id


    with mlflow.start_run(experiment_id=experiment_id) as run:
        data = Data("./dataset/pure_comments.csv",
                    seq_length = SEQUENCE_LENGTH)
        trainingset, testset = random_split(data, [0.8, 0.2])

        train_loader = DataLoader(trainingset, BATCH_SIZE, num_workers=11)
        test_loader = DataLoader(testset, BATCH_SIZE, num_workers=11)

        hyperparameters = {
            "hidden_dim"   : 256,
            "embedding_dim": 128,
            "dropout_rate" : 0.3,
            "learning_rate": 1e-3
        }

        mlflow.log_params(hyperparameters)
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("batch_size", BATCH_SIZE)

        model = Model(**hyperparameters)

        logger = MLFlowLogger(EXPERIMENT_NAME,
                              run_id=run.info.run_id)
        
        T = L.Trainer(logger=logger, max_epochs=3)

        T.fit(model, train_loader, test_loader)



