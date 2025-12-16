from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Turn a time-indexed dataframe into (sequence, target) pairs.

    For each index i, we take:
      X = features[i : i + context_len]
      y = target[i + context_len]

    So the number of usable sequences is:
      len(df) - context_len

    If len(df) <= context_len, the dataset length is 0 and
    the DataLoader will simply yield no batches.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        context_len: int,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.context_len = context_len

        # Store as numpy for speed
        self.features = df[feature_cols].values.astype("float32")
        self.targets = df[target_col].values.astype("float32")

    def __len__(self) -> int:
        n_seq = len(self.targets) - self.context_len
        # Make sure length is never negative
        return max(0, n_seq)

    def __getitem__(self, idx: int):
        # DataLoader will never call this if __len__ == 0
        x = self.features[idx : idx + self.context_len]
        y = self.targets[idx + self.context_len]

        x_tensor = torch.from_numpy(x)  # shape: [context_len, num_features]
        y_tensor = torch.tensor(y, dtype=torch.float32)  # scalar

        return x_tensor, y_tensor