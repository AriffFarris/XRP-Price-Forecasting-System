import torch
import torch.nn as nn


class GRUPredictor(nn.Module):
    """
    GRU-based sequence model:
    Input: (batch, seq_len, input_dim)
    Output: (batch,) next-step prediction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]   # (batch, hidden_dim)
        y = self.fc(last_hidden)      # (batch, 1)
        return y.squeeze(-1)          # (batch,)
