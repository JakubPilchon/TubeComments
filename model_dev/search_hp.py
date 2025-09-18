import mlflow.pytorch
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

EXPERIMENT_NAME = "Comments_tuning"
EXPERIMENT_DESCRIPTION = "Predicting sentiment of youtube comments."

#disable tokenizer parallelism, to avoid deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Current MLflow URI:", mlflow.get_tracking_uri())

mlf_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(),
                              metric_name="val_loss",
                              create_experiment=True)

@mlf_callback.track_in_mlflow()
def objective(trial         : optuna.Trial
              ) -> float:
    
    current_run = mlflow.active_run()
    
    data = Data("./dataset/youtube-comments-sentiment.csv",
                    seq_length = 50)
    trainingset, testset = random_split(data, [0.8, 0.2])

    train_loader = DataLoader(trainingset, 256, num_workers=11)
    val_loader = DataLoader(testset, 256, num_workers=11)
    
        
    hyperparameters = {
            "hidden_dim_1"  : trial.suggest_int("hidden_dim_1", 256, 1024),
            "hidden_dim_2"  : trial.suggest_int("hidden_dim_2", 256, 1024),
            "activation"   : trial.suggest_categorical("activation", ["tanh", "relu"]),
            "embedding_dim": trial.suggest_int("embedding_dim", 512, 2056),
            "dropout_rate" : trial.suggest_float("dropout_rate", 0.1, 0.35),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3)
    }

    mlflow.log_params(hyperparameters)

    model = Model(**hyperparameters)

    ea_call = EarlyStopping(monitor="val_loss",
                                patience=1,
                                mode="min",
                                verbose=False)
        
    #pr_call = PyTorchLightningPruningCallback(trial,
    #                                              monitor="val_loss")
        
    logger = MLFlowLogger(experiment_name="Youtube_Comments_tuning",
                          run_id= current_run.info.run_id)

    trainer = L.Trainer(logger=logger,
                            callbacks=[ea_call],
                            max_epochs=1,
                            devices=[0],
                            #enable_progress_bar=False,  
                            enable_model_summary=False 
                        )
        
    trainer.fit(model, train_loader, val_loader)
    
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.pytorch.log_model(model,
                             name=f"model_{current_run.info.run_name}")

    return trainer.callback_metrics["val_loss"]
    

if __name__ == "__main__":


    mlflow.set_tracking_uri('http://localhost:5000')

    study = optuna.create_study(
                                study_name="Youtube_Comments_tuning",
                                storage="sqlite:///db.sqlite3"
                               )
    study.optimize(objective,
                    n_trials=15,
                    show_progress_bar=True,
                    callbacks=[mlf_callback]
                    )