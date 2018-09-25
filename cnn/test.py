#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 15:20:25 2018

@author: richardolij
"""


import sys;
sys.path.append('../processing/')
import get_data
from get_data import get_dataset

import numpy as np
from CNN_static_basic import CNN

import torch
import torch.optim as optim
import torch.nn as nn
import importlib

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

LEARNING_RATE = 1e-4
BATCH_SIZE = 100
EPOCHS = 20
NUM_CLASSES = 2
EMBEDDING_DIM = 50
SEQUENCE_LENGTH = 500

def get_accuracy(predictions, targets):
    _, pred = torch.max(predictions, 1)
    num_correct = torch.sum(pred == targets, dtype=torch.float, dim = 0)
    accuracy = num_correct / pred.shape[0] * 100
    return accuracy

def train():
    # For colab code update
    importlib.reload(get_data)
    from get_data import get_dataset    

#    dataloader_train, dataloader_validation, dataloader_test= get_datasets(BATCH_SIZE)
    dataloader_train = get_dataset("train", BATCH_SIZE)
    num_batches = len(dataloader_train)
    
    # Initialize the model, optimizer and loss function
    model = CNN(EMBEDDING_DIM, BATCH_SIZE, NUM_CLASSES, SEQUENCE_LENGTH).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    
    sum_loss = 0
    sum_accuracy = 0
    for i in range(EPOCHS):    
        
        for batch_i, data in enumerate(dataloader_train):    
            x, y, doc_ids, doc_lengths = data
            x = torch.tensor(x).to(device)
            x = x.view(-1, EMBEDDING_DIM, SEQUENCE_LENGTH)
            y = torch.tensor(y).long().to(device)
                  
            optimizer.zero_grad()
            outputs = model(x)
            single_loss = loss_function(outputs, y)
            single_loss.backward()
            sum_loss += single_loss
            optimizer.step()
            
            sum_accuracy += get_accuracy(outputs, y)

        accuracy = sum_accuracy / num_batches
        loss = sum_loss / num_batches

        print("Epoch {}, Batch Size = {}, "
              "Accuracy = {:.2f}, Loss = {:.3f}".format(
                i+1, BATCH_SIZE,
                accuracy, loss
        ))
        sum_accuracy = 0
        sum_loss = 0

if __name__ == '__main__':
  
    train()