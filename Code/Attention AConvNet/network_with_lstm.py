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
        self.lstm = nn.LSTM(input_size=1152, hidden_size=512, num_layers=2, batch_first=True)
        self.remaining_layers = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            _blocks.Conv2DBlock(
                shape=[4, 4, 32, self.classes], stride=1, padding='valid',
                w_init=_w_init, b_init=nn.init.zeros_
            ),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten before LSTM
        x, _ = self.lstm(x.unsqueeze(1))  # Adding LSTM layer
        x = x.squeeze(1)
        x = x.view(x.size(0),32, 4, 4)

        x = self.remaining_layers(x)
        return x
