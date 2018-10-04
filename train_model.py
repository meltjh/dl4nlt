import sys;
sys.path.append('processing/')
import get_data
from get_data import get_dataset

import lstm_cnn
import cnn
import lstm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

MODEL = "CNN" # "CNN", "LSTM", "LSTM_CNN"
LEARNING_RATE = 0.005
BATCH_SIZE = 64
EPOCHS = 40
NUM_CLASSES = 2
WORD_EMBEDDING_DIM = 50
SEQUENCE_LENGTH = 500
NUM_HIDDEN = 256
NUM_FEATUREMAPS = 512
NUM_LSTM_LAYERS = 2
DROP_OUT = 0.5
REGULARISATION = 0.001

print("Learning rate", LEARNING_RATE)

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
    hyper_parameters = str(MODEL) + "lr" + str(LEARNING_RATE) + "_batchsize" + \
        str(BATCH_SIZE) + "_embeddim" + str(WORD_EMBEDDING_DIM) + "_hidden" + \
        str(NUM_HIDDEN) + "_layers" + str(NUM_LSTM_LAYERS) + "_dropout" + str(DROP_OUT) +"_regularisation" + str(REGULARISATION)
    file_name = folder + "/" + hyper_parameters + "_checkpoint" + \
        str(epoch) + ".pth"

    if not os.path.isdir(folder):
        os.mkdir(folder)

    state = {"state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict()}

    torch.save(state, file_name)
    
def plot_results(loss_history, accuracy_history_train, accuracy_history_validation):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy %', color='C0')
    ax1.tick_params('y', colors='C0')
    
    ax1.plot(accuracy_history_train, color='C0', linestyle=":", label="Train")
    ax1.plot(accuracy_history_validation, color='C0', linestyle='--', label="Validation")
    
    ax1.legend()
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='C1')
    ax2.tick_params('y', colors='C1')
    
    ax2.plot(loss_history, color='C1', linestyle="-", label="Loss")
    
    fig.tight_layout()
    plt.title = MODEL
    plt.show()


def train():
    """
    Trains the model.
    """

    dataloader_train = get_dataset("train", BATCH_SIZE)
    num_batches = len(dataloader_train)
    
    dataloader_validation = get_dataset("validation", BATCH_SIZE)

    # Initialize the model, optimizer and loss function
    if MODEL == "CNN":
        model = cnn.CNN(WORD_EMBEDDING_DIM, NUM_FEATUREMAPS, NUM_CLASSES, SEQUENCE_LENGTH, DROP_OUT).to(device)
        print("MODEL {}, WORD_EMBEDDING_DIM {}, NUM_FEATUREMAPS {}, NUM_CLASSES {}, SEQUENCE_LENGTH {}, DROP_OUT {}, REGULARISATION {}".format(MODEL, WORD_EMBEDDING_DIM, NUM_FEATUREMAPS, NUM_CLASSES, SEQUENCE_LENGTH, DROP_OUT, REGULARISATION))
    elif MODEL == "LSTM":
        model = lstm.LSTM(WORD_EMBEDDING_DIM, NUM_CLASSES, NUM_HIDDEN, NUM_LSTM_LAYERS, device).to(device)
        print("MODEL {}, WORD_EMBEDDING_DIM {}, NUM_CLASSES {}, NUM_HIDDEN {}, NUM_LSTM_LAYERS {}, REGULARISATION {}".format(MODEL, WORD_EMBEDDING_DIM, NUM_CLASSES, NUM_HIDDEN, NUM_LSTM_LAYERS, REGULARISATION))
    elif MODEL == "LSTM_CNN":
        model = lstm_cnn.LSTM_CNN(WORD_EMBEDDING_DIM, NUM_CLASSES, NUM_HIDDEN, NUM_LSTM_LAYERS, NUM_FEATUREMAPS, SEQUENCE_LENGTH, DROP_OUT).to(device)
        print("MODEL {}, WORD_EMBEDDING_DIM {}, NUM_CLASSES {}, NUM_HIDDEN {}, NUM_LSTM_LAYERS {}, NUM_FEATUREMAPS {}, SEQUENCE_LENGTH {}, REGULARISATION {}, DROP_OUT{}".format(MODEL, WORD_EMBEDDING_DIM, NUM_CLASSES, NUM_HIDDEN, NUM_LSTM_LAYERS, NUM_FEATUREMAPS, SEQUENCE_LENGTH, REGULARISATION, DROP_OUT))
    else:
        raise NotImplementedError("Model {} does not exist".format(MODEL))
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARISATION)
    loss_function = nn.CrossEntropyLoss()

    ### LOAD MODEL IF YOU WANT TO ###
    # checkpoint = torch.load('model_states/checkpoint1.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    loss_history = []
    accuracy_history_train = []
    accuracy_history_test = []

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

        # Epoch done, print results
        # Test on validation set
        sum_accuracy_validation = 0.0
        for validation_batch in dataloader_validation:
            x_val, y_val, doc_ids_val, doc_lengths_val = validation_batch
            x_val = torch.tensor(x_val).to(device)
            y_val = torch.tensor(y_val).long().to(device)
            outputs_val = model.forward(x_val, doc_lengths_val, False)
            sum_accuracy_validation += get_accuracy(outputs_val, y_val)
        accuracy_validation = sum_accuracy_validation / len(dataloader_validation)
        
        # Average training results
        accuracy = sum_accuracy / num_batches
        loss = sum_loss / num_batches

        print("Epoch {}, Loss = {:.3f}, Train accuracy = {:.2f}, Validation accuracy = {:.2f}".format(
                i+1, loss, accuracy, accuracy_validation))
        
        # Append results for plot
        loss_history.append(loss)
        accuracy_history_train.append(accuracy)
        accuracy_history_test.append(accuracy_validation)
        
        # Reset the results for next epoch
        sum_accuracy = 0
        sum_loss = 0

        # Save model
        save_checkpoint(model, optimizer, i)

    # Plot the results when all epochs are done
    plot_results(loss_history, accuracy_history_train, accuracy_history_test)

if __name__ == '__main__':

    train()
