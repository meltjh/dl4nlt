import sys;
sys.path.append('../processing/')
from get_data import get_datasets

import numpy as np
from lstm import LSTM

import torch
import torch.optim as optim
import torch.nn as nn

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

LEARNING_RATE = 1e-4
BATCH_SIZE = 300
EPOCHS = 2

def get_accuracy(predictions, targets):
    _, pred = torch.max(predictions, 1)
    num_correct = torch.sum(pred == targets, dtype=torch.float, dim = 0)
    accuracy = num_correct / pred.shape[0] * 100
    return accuracy


def train():
    # dataloader_train, _, _ = get_datasets(BATCH_SIZE)
    dataloader_train = get_datasets(BATCH_SIZE)


    # Initialize the model, optimizer and loss function
    model = LSTM(50, BATCH_SIZE, 2, 128, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    sum_loss = 0
    sum_accuracy = 0

    for i in range(EPOCHS):

        for batch_i, data in enumerate(dataloader_train):
            x, y, doc_ids, doc_lengths = data
            seq_length = x.shape[1]

            x = torch.tensor(x).to(device)
            x = x.view(seq_length, BATCH_SIZE, -1)
            y = torch.tensor(y).long().to(device)

            optimizer.zero_grad()
            outputs = model(x)
            single_loss = loss_function(outputs, y)
            single_loss.backward()
            sum_loss += single_loss
            optimizer.step()

            sum_accuracy += get_accuracy(outputs, y) # fixme

            if batch_i % 10 == 0:
                accuracy = sum_accuracy / 10
                loss = sum_loss / 10

                print("Train Step {}, Batch Size = {}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        batch_i + i * EPOCHS, BATCH_SIZE,
                        accuracy, loss
                ))
                sum_accuracy = 0
                sum_loss = 0

if __name__ == '__main__':

    train()
