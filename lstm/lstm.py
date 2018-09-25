from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_size, batch_size, output_size, num_hidden,
                 num_layers):

        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, num_hidden, num_layers)
        self.linear = nn.Linear(num_hidden, output_size)


    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.linear(output[-1, :, :])

        return output
