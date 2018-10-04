from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_CNN(nn.Module):
    
    def __init__(self, input_size, output_size, num_hidden,
                 num_layers, n_featuremaps, seq_length, dropout):

        super(LSTM_CNN, self).__init__()

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        
        self.lstm = nn.LSTM(input_size, num_hidden, num_layers, batch_first=True)
        
        self.conv1_3 = nn.Conv1d(num_hidden, n_featuremaps, 3)
        self.conv1_4 = nn.Conv1d(num_hidden, n_featuremaps, 4)
        self.conv1_5 = nn.Conv1d(num_hidden, n_featuremaps, 5)
    
        self.maxpool1_3 = nn.MaxPool1d(seq_length-3+1)
        self.maxpool1_4 = nn.MaxPool1d(seq_length-4+1)
        self.maxpool1_5 = nn.MaxPool1d(seq_length-5+1)
        
        self.linear_1 = nn.Linear(3*n_featuremaps, output_size)
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, lengths, is_train = True):
        
        # LSTM
        output, _ = self.lstm(x)

        x = output.permute(0,2,1)
        
        # CNN architecture from Yoon Kim
        x_3 = F.relu_(self.conv1_3(x))
        x_4 = F.relu_(self.conv1_4(x))
        x_5 = F.relu_(self.conv1_5(x))
        
        # Extract the most relevant feature from each feature map.
        x_3 = self.maxpool1_3(x_3)
        x_4 = self.maxpool1_4(x_4)
        x_5 = self.maxpool1_5(x_5)
        
        # Concatenate the most relevant features.
        cat_3_4 = torch.cat((x_3, x_4), 1)
        cat_3_4_5 = torch.cat((cat_3_4, x_5), 1)
        pen_layer = torch.squeeze(cat_3_4_5, 2)
        
        output = self.linear_1(pen_layer)
    
        # Apply drop-out during training.
        if is_train:
            output = self.dropout(output)
    
        return output 