"""Dataset assembly: supervised frames, walk-forward splits, scaling, windowing."""

from berich.datasets.assemble import SupervisedDataset, build_dataset
from berich.datasets.cross_sectional import PanelDataset, build_panel_dataset
from berich.datasets.scaling import StandardScaler
from berich.datasets.splits import Fold, walk_forward_splits
from berich.datasets.windows import make_sequences

__all__ = [
    "Fold",
    "PanelDataset",
    "StandardScaler",
    "SupervisedDataset",
    "build_dataset",
    "build_panel_dataset",
    "make_sequences",
    "walk_forward_splits",
]
