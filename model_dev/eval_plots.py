import polars as pl
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from lightning import Callback
from typing import Callable, Any, Tuple, Iterable
from matplotlib.figure import Figure
import mlflow

rev_onehot = {
            0: "Positive",
            1: "Neutral",
            2: "Negative"
        }

class PlotCallback(Callback):
    def __init__(self,
                 funcs: Iterable[Callable[[pl.DataFrame], Figure]] = None,
                 test_dataset :torch.utils.data.Subset= None
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
            fig = f(self.df)



def plot_conf_matrix(
                    figsize: Tuple[float, float] = (8, 6.5),
                    cmap : str = "flare"
                    ) -> Callable[[pl.DataFrame], Figure]:
    

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
        mlflow.log_figure(fig, "confusion_matrix.png")
        return fig
    
    return func

    