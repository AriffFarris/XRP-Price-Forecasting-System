from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Sliding window time-series dataset.

    For each index i:
      X = df[feature_cols][i : i+context_len]
      y = df[target_col][i+context_len]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
        context_len: int = 72,
    ):
        self.df = df
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.context_len = context_len

        self.features = df[self.feature_cols].values.astype("float32")
        self.targets = df[self.target_col].values.astype("float32")

        self.n = len(df)
        self.max_start = self.n - context_len - 1

    def __len__(self) -> int:
        return self.max_start

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.context_len

        x = self.features[start:end]   # (context_len, num_features)
        y = self.targets[end]          # scalar

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.tensor(y)
        return x_tensor, y_tensor