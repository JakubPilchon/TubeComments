import polars as pl
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from lightning import Callback
from typing import Callable, Any
from matplotlib.figure import Figure

rev_onehot = {
            0: "Positive",
            1: "Neutral",
            2: "Negative"
        }

class PlotCallback(Callback):
    def __init__(self,
                 funcs: Callable[[pl.DataFrame], Figure] = None,
                 test_dataset :torch.utils.data.Subset= None
                ):
        self.df : pl.DataFrame = test_dataset.dataset.dataframe[test_dataset.indices]
        self.predictions = []
        super().__init__()

    

    def on_fit_end(self, trainer, pl_module):
        predictions = trainer.predict(pl_module, trainer.val_dataloaders)
        predictions = list(map(rev_onehot.get,
                               torch.cat(predictions, dim=0).reshape(-1).tolist()))
        
        assert len(predictions) == len(self.df)
        self.df = self.df.with_columns(
            pl.Series("predictions", predictions)
        )
        print(self.df.head())

def plot_conf_matrix(
                     df: pl.DataFrame
                    ) -> Figure:
    
    with plt.style.context(style="ggplot"):
        ax, fig = plt.subplots()

    