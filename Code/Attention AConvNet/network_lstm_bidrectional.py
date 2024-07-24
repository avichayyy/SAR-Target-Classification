import torch.nn as nn
import torch

from . import _blocks


class Network(nn.Module):

    def __init__(self, **params):
        super(Network, self).__init__()
        self.dropout_rate = params.get('dropout_rate', 0.5)
        self.classes = params.get('classes', 10)
        self.channels = params.get('channels', 1)

        _w_init = params.get('w_init', lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu'))
        _b_init = params.get('b_init', lambda x: nn.init.constant_(x, 0.1))

        self.conv_blocks = nn.Sequential(
            _blocks.Conv2DBlock(
                shape=[5, 5, self.channels, 16], stride=1, padding='valid', activation='relu', max_pool=True,
                w_init=_w_init, b_init=_b_init
            ),
            _blocks.Conv2DBlock(
                shape=[5, 5, 16, 32], stride=1, padding='valid', activation='relu', max_pool=True,
                w_init=_w_init, b_init=_b_init
            ),
            _blocks.Conv2DBlock(
                shape=[6, 6, 32, 64], stride=1, padding='valid', activation='relu', max_pool=True,
                w_init=_w_init, b_init=_b_init
            ),
            _blocks.Conv2DBlock(
                shape=[5, 5, 64, 128], stride=1, padding='valid', activation='relu',
                w_init=_w_init, b_init=_b_init
            ),
        )
        self.lstm = nn.LSTM(input_size=1152, hidden_size=64, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(64, self.classes)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten before LSTM
        x, _ = self.lstm(x.unsqueeze(1))  # Adding LSTM layer
        x = x.squeeze(1)
        x = x.view(x.size(0), -1)  # Flatten LSTM output

        x = self.fc(x)
        return x