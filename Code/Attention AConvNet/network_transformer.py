import torch.nn as nn
import torch

from . import _blocks


class Network(nn.Module):

    def __init__(self, **params):
        super(Network, self).__init__()
        self.dropout_rate = params.get('dropout_rate', 0.5)
        self.classes = params.get('classes', 10)
        self.channels = params.get('channels', 1)
        self.input_size = params.get('input_size', 128)

        _w_init = params.get('w_init', lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu'))
        _b_init = params.get('b_init', lambda x: nn.init.constant_(x, 0.1))

        # Assuming you have defined Conv2DBlock and Flatten as before
        self.conv_blocks = nn.Sequential(
            _blocks.Conv2DBlock(shape=[5, 5, self.channels, 16], stride=1, padding='valid', activation='relu',
                w_init=_w_init, b_init=_b_init),
            _blocks.Conv2DBlock(shape=[5, 5, 16, 32], stride=1, padding='valid', activation='relu', max_pool=True,
                w_init=_w_init, b_init=_b_init),
            _blocks.Conv2DBlock(shape=[6, 6, 32, 64], stride=1, padding='valid', activation='relu', max_pool=True,
                w_init=_w_init, b_init=_b_init),
            _blocks.Conv2DBlock(shape=[5, 5, 64, 128], stride=1, padding='valid', activation='relu',
                w_init=_w_init, b_init=_b_init)
        )

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.input_size, nhead=16),
            num_layers=1
        )

        self.remaining_layers = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            _blocks.Conv2DBlock(shape=[3, 3, 128, self.classes], stride=1, padding='valid', activation='relu',
                w_init=_w_init, b_init=nn.init.zeros_),
            nn.Flatten()
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv_blocks(x)
        b, c, h, w = x.size()  # Get dimensions after convolutions
        
        # Reshape to match transformer input requirements
        x = x.view(b, c, h * w)  # Assuming input is flattened or reshaped
        
        # Transformer Encoder expects (seq_length, batch_size, input_size)
        x = x.permute(2, 0, 1)  # Reshape to (seq_length, batch_size, input_size)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Reshape back to (batch_size, channels, height, width)
        x = x.permute(1, 2, 0)  # Reshape to (batch_size, channels, seq_length)
        x = x.view(b, c, h, w)  # Reshape to (batch_size, channels, height, width)
        
        # Remaining layers
        x = self.remaining_layers(x)
        
        return x