import sys;
sys.path.append('../processing/')
import get_data
from get_data import get_dataset

import numpy as np
from lstm_gather import LSTM

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import importlib
import argparse

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

LEARNING_RATE = 1e-4 # geprobeerd 1e-2, 1e-4
BATCH_SIZE = 64 # geprobeerd, 32, ..., 640
EPOCHS = 20
NUM_CLASSES = 2
EMBEDDING_DIM = 50
SEQUENCE_LENGTH = 500
NUM_HIDDEN = 256 # geprobeerd 32, 128
NUM_LAYERS = 1

def get_accuracy(predictions, targets):
    _, pred = torch.max(predictions, 1)
    num_correct = torch.sum(pred == targets, dtype=torch.float, dim = 0)
    accuracy = num_correct / pred.shape[0] * 100
    return accuracy

def save_checkpoint(state, epoch):
    ''' Save the model ''' 
    file_name = 'model_states/checkpoint' + str(epoch) + '.pth'
    torch.save(state, file_name)


def train():
    # For colab code update
    importlib.reload(get_data)
    from get_data import get_dataset

    dataloader_train = get_dataset("train", BATCH_SIZE)
    num_batches = len(dataloader_train)

    # Initialize the model, optimizer and loss function
    model = LSTM(EMBEDDING_DIM, BATCH_SIZE, NUM_CLASSES, NUM_HIDDEN,
                 NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    ### LOAD MODEL IF YOU WANT TO ### 
    # checkpoint = torch.load('model_states/checkpoint1.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    sum_loss = 0
    sum_accuracy = 0

    for i in range(EPOCHS):

        for batch_i, data in enumerate(dataloader_train):
            x, y, doc_ids, doc_lengths = data

            x = torch.tensor(x).to(device)
            y = torch.tensor(y).long().to(device)

            optimizer.zero_grad()

            outputs = model(x, doc_lengths)

            single_loss = loss_function(outputs, y)
            single_loss.backward()

            # print(single_loss.item())

            sum_loss += float(single_loss)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25) # Nog niet zeker wat dit doet, maar iig tegen vanishing / exploding gradient, ps, werkt niet :'(

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
        save_checkpoint({'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict()}, i)

if __name__ == '__main__':

    train()
