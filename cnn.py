"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
  """
  This is the CNN architecture as designed by Yoon Kim. 
  Given the embedded text, the model applies three different filter sizes.
  This gives n_featuremaps per filter size. By using maxpooling, the most
  relevant features are extracted and mapped to a linear layer. 
  """

  def __init__(self, n_channels, n_featuremaps, n_classes, seq_length, dropout):
    """
    Initializes ConvNet object.
    
    """
    
    super(CNN, self).__init__()    
    self.conv1_3 = nn.Conv1d(n_channels, n_featuremaps, 3)
    self.conv1_4 = nn.Conv1d(n_channels, n_featuremaps, 4)
    self.conv1_5 = nn.Conv1d(n_channels, n_featuremaps, 5)

    self.maxpool1_3 = nn.MaxPool1d(seq_length-3+1)
    self.maxpool1_4 = nn.MaxPool1d(seq_length-4+1)
    self.maxpool1_5 = nn.MaxPool1d(seq_length-5+1)
    
    self.linear_1 = nn.Linear(3*n_featuremaps, n_classes)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, lengths, is_train = True):
    x = torch.transpose(x, 1, 2)
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
    
    out = self.linear_1(pen_layer)
    
    # Apply drop-out during training.
    if is_train:
        out = self.dropout(out)

    return out