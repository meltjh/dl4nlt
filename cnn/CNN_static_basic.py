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
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_featuremaps, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    super(CNN, self).__init__()

    self.conv1_3 = nn.Conv1d(n_channels, n_featuremaps, 3)
    self.conv1_4 = nn.Conv1d(n_channels, n_featuremaps, 4)
    self.conv1_5 = nn.Conv1d(n_channels, n_featuremaps, 5)
    
    
    self.maxpool1_3 = nn.MaxPool1d(n_featuremaps)
    self.maxpool1_4 = nn.MaxPool1d(n_featuremaps)
    self.maxpool1_5 = nn.MaxPool1d(n_featuremaps)
    
    self.linear_1 = nn.Linear(3, n_classes)
    
 
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    print("x",x.shape)

    x_3 = F.relu_(self.conv1_3(x))
    x_4 = F.relu_(self.conv1_4(x))
    x_5 = F.relu_(self.conv1_5(x))
    
    print("x_3",x.shape)
    
    x_3 = self.maxpool1_3(x_3)
    x_4 = self.maxpool1_4(x_4)
    x_5 = self.maxpool1_5(x_5)
    
    print("x_3",x_3.shape)
    
    # Moet -1*300 worden (3*100 filters)
    raise NotImplementedError
    
    x =  self.linear_1()
    
    
#    out = F.softmax(x, dim=1)
    out = x

    ########################
    # END OF YOUR CODE    #
    #######################

    return out
