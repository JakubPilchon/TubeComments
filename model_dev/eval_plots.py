import polars as pl
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lightning import Callback
from typing import Callable, Any, Tuple, Iterable
from matplotlib.figure import Figure
from transformers import AutoTokenizer, Trainer
from datasets import Dataset
import mlflow

type PlotFunction = Callable[[pl.DataFrame], Tuple[Figure, str]]

rev_onehot = {
            0: "Positive",
            1: "Neutral",
            2: "Negative"
        }

class PlotCallback(Callback):
    def __init__(self,
                 funcs: Iterable[PlotFunction],
                 test_dataset :torch.utils.data.Subset
                ):
        self.funcs = funcs
        self.df : pl.DataFrame = test_dataset.dataset.dataframe[test_dataset.indices]
        self.predictions = []
        super().__init__()

    def on_fit_end(self, trainer, pl_module):
        predictions = trainer.predict(pl_module, trainer.val_dataloaders)
        predictions = list(map(rev_onehot.get,
                               torch.cat(predictions, dim=0).reshape(-1).tolist()))
        
        assert len(predictions) == len(self.df)
        self.df = self.df.with_columns(
            pl.Series("Predicted", predictions)
        )

        for f in self.funcs:
            fig, text = f(self.df)
            mlflow.log_figure(fig, text + ".png")

def transformerPlot(
        trainer: Trainer,
        dataset: Dataset,
        funcs: Iterable[PlotFunction] = None
                    ) -> None:
    
    id2label = {
             0 : "Positive",
             1 : "Neutral",
             2 : "Negative"
        }
    
    results = trainer.predict(dataset)

    print(np.argmax(results.predictions, axis=1))
    df = (dataset
            .to_polars()
            .rename({
                "text":"CommentText",
                "labels":"Sentiment"
            })
            .with_columns(
                pl.lit(np.argmax(results.predictions, axis=1))
                .replace(id2label, default=None)
                .alias("Predicted"),
                pl.col("Sentiment")
                .replace(id2label, default=None)
            ))

    for f in funcs:
            fig, text = f(df)
            mlflow.log_figure(fig, text + ".png")



def plot_conf_matrix(
                    figsize: Tuple[float, float] = (8, 6.5),
                    cmap : str = "flare"
                    ) -> PlotFunction:
    

    def func(df : pl.DataFrame
             ) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)


        sns.heatmap((df
            .pivot(on="Sentiment",
                    index="Predicted",
                    values="Sentiment",
                    aggregate_function="len")
            .select(("Predicted", "Positive", "Neutral", "Negative"))
            .sort(by="Predicted",
                  descending=True)
            .to_pandas()
            .set_index("Predicted")
            .div(len(df), axis=0)),
        annot=True,
        cmap=cmap,
        ax = ax)

        ax.set_title("Confusion Matrix")
        return (fig, "confusion_matrix")
    
    return func

def plot_class_accuracy(
                    figsize: Tuple[float, float] = (8, 6.5)
                    ) -> PlotFunction:
    

    def func(df : pl.DataFrame
             ) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)

        sns.barplot(
            (df
                .with_columns((pl.col("Sentiment") == pl.col("Predicted")).alias("Correct"))
                .group_by("Sentiment")
                .agg(pl.col("Correct").mean())
            ),
            y="Correct",
            x="Sentiment",
            ax=ax)
        
        ax.set(ylim=(0, 1),
              ylabel="Accuracy",
              xlabel="Sentiment")
        
        for container in ax.containers:
            ax.bar_label(container)
        
        ax.set_title("Accuracy by Sentiment")
        return (fig, "class_accuracy")
    
    return func

def plot_length_accuracy(
                tokenizer: AutoTokenizer,
                figsize: Tuple[float, float] = (8, 6.5),
                filter_size: float = 2.,
                alpha: float = .5                         
                ) -> PlotFunction:
    
    def func(df: pl.DataFrame) -> Figure:
        lengths = list(map(len, tokenizer(df["CommentText"].to_list())["input_ids"]))

        df = (df
          .with_columns(
              pl.Series(lengths).alias("TokenLength"),
              (pl.col("Sentiment") == pl.col("Predicted")).alias("Correct")
              )
          .filter(
              pl.col("TokenLength") < pl.col("TokenLength").mean() + filter_size * pl.col("TokenLength").std()
              )
            )
        
        fig, ax = plt.subplots(figsize=figsize)

        sns.violinplot(
            x=df["TokenLength"].to_numpy(),
            hue = df["Correct"].to_numpy(),
            fill=True,
            alpha=alpha,
            ax=ax
        )

        ax.set_title("Distribution of Length by correct prediction")
        return (fig, "length_accuracy")
    
    return func