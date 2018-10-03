from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class LSTM(nn.Module):
    
    def __init__(self, input_size, output_size, num_hidden,
                 num_layers, device):

        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.device = device
        self.lstm = nn.LSTM(input_size, num_hidden, num_layers, batch_first=True)
        self.linear = nn.Linear(num_hidden, output_size)


    def forward(self, x, lengths, is_train = True):
        batch_size = x.shape[0]
        # max review lengths 
        max_lengths = torch.LongTensor(lengths.long()).to(self.device)
        output, _ = self.lstm(x)
        # expand batch size x 1 x num_units (units in hidden layer)
        last_out = torch.gather(output, 1, max_lengths.view(-1,1,1).expand(batch_size,1,self.num_hidden)-1)
        last_out = torch.squeeze(last_out, 1) # last out returns batch_size x 1 x num_hidden
        output = self.linear(last_out)

        return output 