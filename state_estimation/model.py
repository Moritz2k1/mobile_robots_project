from torch import nn, split, cat
import numpy as np
from matplotlib import pyplot as plt

class LSTM(nn.Module):

    def __init__(self, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.lstm_x = nn.ModuleDict({
            'lstm': nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            ),
            'linear': nn.Linear(hidden_size, 1)
        })
        self.lstm_y = nn.ModuleDict({
            'lstm': nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            ),
            'linear': nn.Linear(hidden_size, 1)
        })
        self.lstm_yaw = nn.ModuleDict({
            'lstm': nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            ),
            'linear': nn.Linear(hidden_size, 1)
        })

    def forward(self, x):
        x_val, y_val, yaw_val = split(x, 1, dim=-1)

        x_val, _ = self.lstm_x['lstm'](x_val)
        x_val = self.lstm_x['linear'](x_val)
        x_val = x_val[:, -1, :]  # Select last time step

        y_val, _ = self.lstm_y['lstm'](y_val)
        y_val = self.lstm_y['linear'](y_val)
        y_val = y_val[:, -1, :]

        yaw_val, _ = self.lstm_yaw['lstm'](yaw_val)
        yaw_val = self.lstm_yaw['linear'](yaw_val)
        yaw_val = yaw_val[:, -1, :]

        return cat([x_val, y_val, yaw_val], dim=-1)  # Shape: [batch_size, 3]

