#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 15:20:25 2018

@author: richardolij
"""


import sys;
sys.path.append('../processing/')
from get_data import get_datasets

import numpy as np
from CNN_static_basic import CNN

import torch
import torch.optim as optim
import torch.nn as nn

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

LEARNING_RATE = 1e-4
BATCH_SIZE = 5
EPOCHS = 2



def train():
    #  dataloader_train, dataloader_validation, dataloader_test= get_datasets(BATCH_SIZE)
    
    # Initialize the model, optimizer and loss function
    convnet = CNN(1, 100, 2).to(device)
    optimizer = optim.Adam(convnet.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    
    sum_loss = 0
    for i in range(EPOCHS):    
        
        for idx, data in enumerate(dataloader_train):    
            x, y, doc_ids, doc_lengths = data
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)
                  
            optimizer.zero_grad()
            predictions = convnet.forward(x)
            loss = loss_function(predictions, y)
            loss.backward()
            optimizer.step()
            
            sum_loss += float(loss)
            
        print("sum_loss batch {}, {}".format(i, sum_loss))
          
          


if __name__ == '__main__':
  
    train()