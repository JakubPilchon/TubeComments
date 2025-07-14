import mlflow.pytorch
import torch 
import lightning as L
import mlflow 
import optuna
import os
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from optuna.integration.mlflow import MLflowCallback
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from dataset import Data
from model import Model
from functools import partial

EXPERIMENT_NAME = "youtube_comments"
EXPERIMENT_DESCRIPTION = "Predicting sentiment of youtube comments."

#disable tokenizer parallelism, to avoid deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def objective(trial         : optuna.Trial,
              train_loader  : DataLoader,
              val_loader    : DataLoader,
              experiment_id : str
              ) -> float:
    
    with mlflow.start_run(nested=True, experiment_id=experiment_id) as run:

        mlflow.set_tag("type", "optuna run")
        
        hyperparameters = {
            "hidden_dim_1"  : trial.suggest_int("hidden_dim_1", 32, 512),
            "hidden_dim_2"  : trial.suggest_int("hidden_dim_2", 32, 512),
            "activation"   : trial.suggest_categorical("activation", ["tanh", "relu"]),
            "embedding_dim": trial.suggest_int("embedding_dim", 64, 256),
            "dropout_rate" : trial.suggest_float("dropout_rate", 0.1, 0.35),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3)
        }

        mlflow.log_params(hyperparameters)

        model = Model(**hyperparameters)

        ea_call = EarlyStopping(monitor="val_loss",
                                patience=2,
                                mode="min",
                                verbose=False)
        
        pr_call = PyTorchLightningPruningCallback(trial,
                                                  monitor="val_loss")
        
        logger = MLFlowLogger(EXPERIMENT_NAME,
                              run_name=run.info.run_name)
        
        trainer = L.Trainer(logger=logger,
                            callbacks=[ea_call, pr_call],
                            max_epochs=10,
                            enable_progress_bar=False,  
                            enable_model_summary=False 
                        )
        
        trainer.fit(model, train_loader, val_loader)

        return trainer.callback_metrics["val_loss"]
    

if __name__ == "__main__":

    cli = mlflow.MlflowClient()
    experiment = cli.get_experiment_by_name(EXPERIMENT_NAME) # we assume that the experiment already exists


    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        data = Data("./dataset/pure_comments.csv",
                    seq_length = 100)
        trainingset, testset = random_split(data, [0.8, 0.2])

        train_loader = DataLoader(trainingset, 256, num_workers=11)
        test_loader = DataLoader(testset, 256, num_workers=11)

        objective = partial(objective,
                            train_loader=train_loader,
                            val_loader=test_loader,
                            experiment_id = experiment.experiment_id)
        
        study = optuna.create_study()
        study.optimize(objective,
                    n_trials=10,
                    show_progress_bar=True)
        
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best validation loss",study.best_value)