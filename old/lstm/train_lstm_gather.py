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
import os

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
NUM_LAYERS = 2


def get_accuracy(predictions, targets):
    """
    Calculates the accuracy.
    """
    _, pred = torch.max(predictions, 1)
    num_correct = torch.sum(pred == targets, dtype=torch.float, dim = 0)
    accuracy = num_correct / pred.shape[0] * 100
    return accuracy


def save_checkpoint(model, optimizer, epoch):
    """
    Saves the model.
    """

    folder = "model_states"
    hyper_parameters = "lr" + str(LEARNING_RATE) + "_batchsize" + \
        str(BATCH_SIZE) + "_embeddim" + str(EMBEDDING_DIM) + "_hidden" + \
        str(NUM_HIDDEN) + "_layers" + str(NUM_LAYERS)
    file_name = folder + "/" + hyper_parameters + "_checkpoint" + \
        str(epoch) + ".pth"

    if not os.path.isdir(folder):
        os.mkdir(folder)

    state = {"state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict()}

    torch.save(state, file_name)


def train():
    """
    Trains the model.
    """

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

            sum_loss += float(single_loss)

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

        save_checkpoint(model, optimizer, i)


if __name__ == '__main__':

    train()
