"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""


#from read_data import get_data
import sys;
sys.path.append('../data_preprocessing/')
import read_data



import numpy as np
from convnet import ConvNet

import torch
import torch.optim as optim
import torch.nn as nn

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

# Default constants
LEARNING_RATE = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'



def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  
  data_loader = read_data.get_data()

  # Initialize the model, optimizer and loss function
  convnet = ConvNet(1, 2).to(device)
  optimizer = optim.Adam(convnet.parameters(), lr=LEARNING_RATE)
  loss_function = nn.CrossEntropyLoss()

  epochs = 2
  
  for i in range(epochs):
      
      for idx, data in enumerate(data_loader):
      
          x, y = data
          
          print(x.shape)
          print(y.shape)
          break
          
          
          x = torch.tensor(x).to(device)
          y = torch.tensor(y).to(device)
          
          # Only get the indices rather than the one-hot vectors.
          y = y.max(-1)[1]
                
          optimizer.zero_grad()
          predictions = convnet.forward(x)
          loss = loss_function(predictions, y)
          loss.backward()
          optimizer.step()
          
          total_loss += float(loss)
          total_batch_loss += float(loss)
          total_batch_step += 1
          step += 1


if __name__ == '__main__':
  
    train()