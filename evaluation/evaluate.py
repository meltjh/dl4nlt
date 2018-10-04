import torch

import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import re

import sys
sys.path.append('../processing')
import preprocess_data
from processing.get_data import get_dataset
sys.path.append('../')
from lstm import LSTM
from cnn import CNN
from lstm_cnn import LSTM_CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = "LSTM" # "CNN", "LSTM", "LSTM_CNN"
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 40
NUM_CLASSES = 2
WORD_EMBEDDING_DIM = 50
SEQUENCE_LENGTH = 500
DROP_OUT = 0.5
REGULARISATION = 0.001

# For LSTM
NUM_LSTM_LAYERS = 2
NUM_HIDDEN = 256

# For CNN
NUM_FEATUREMAPS = 512

def get_model(file_name):
    """
    Loads the model and optimizer.
    """
    if MODEL == "LSTM":
        model = LSTM(WORD_EMBEDDING_DIM, NUM_CLASSES, NUM_HIDDEN, NUM_LSTM_LAYERS, device).to(device)
    elif MODEL == "CNN":
        model = CNN(WORD_EMBEDDING_DIM, NUM_FEATUREMAPS, NUM_CLASSES, SEQUENCE_LENGTH, DROP_OUT).to(device)
    else:
        model = LSTM_CNN(WORD_EMBEDDING_DIM, NUM_CLASSES, NUM_HIDDEN, NUM_LSTM_LAYERS, NUM_FEATUREMAPS, SEQUENCE_LENGTH).to(device)

    saved = torch.load(file_name, map_location="cpu")
    model.load_state_dict(saved["state_dict"])
    print("Loaded model")
    return model


def evaluate(model, dataloader):
    """
    Evaluates the model by calculating the accuracy, precision and recall.
    """

    num_batches = len(dataloader)
    sum_accuracy = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for batch_i, data in enumerate(dataloader):
            x, y, doc_ids, doc_lengths = data

            x = torch.tensor(x).to(device)
            y = torch.tensor(y).long().to(device)

            outputs = model(x, doc_lengths, False)
            outputs = torch.max(outputs, 1)[1]

            accuracy = get_accuracy(outputs, y)
            sum_accuracy += accuracy.item()

            tp, fp, fn = get_TP_FP_FN(outputs, y)
            true_positives += tp
            false_positives += fp
            false_negatives += fn

    precision, recall = get_precision_recall(true_positives, false_positives,
                                             false_negatives)
    return sum_accuracy/num_batches, precision, recall


def get_accuracy(predictions, targets):
    """
    Calculates the accuracy.
    """
    num_correct = torch.sum(predictions == targets, dtype=torch.float, dim=0)
    accuracy = num_correct / predictions.shape[0] * 100
    return accuracy


def get_TP_FP_FN(predictions, targets):
    """
    Calculates the true positives, false positives and false negatives between
    a set of predictions and targets.
    """

    true = torch.ones(predictions.shape)
    false = torch.zeros(predictions.shape)

    true_positives = torch.sum(torch.where((targets==1) & (predictions==1),
                                           true, false)).item()
    false_positives = torch.sum(torch.where((targets==0) & (predictions==1),
                                            true, false)).item()
    false_negatives = torch.sum(torch.where((targets==1) & (predictions==0),
                                            true, false)).item()
    return true_positives, false_positives, false_negatives


def get_precision_recall(true_positives, false_positives, false_negatives):
    """
    Calculates the precision and recall.
    """
    print("Calculating precision and recall")
    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)
    return precision, recall

if __name__ == "__main__":
    test_loader = get_dataset("test", BATCH_SIZE)
    file_name = "final_models/LSTMlr0.005_batchsize64_embeddim50_hidden256_layers2_dropout0.5_regularisation0.001_checkpoint22.pth"
    model = get_model(file_name)
    accuracy, precision, recall = evaluate(model, test_loader)
    print("File:", file_name)
    print("Accuracy:", str(accuracy))
    print("Precision:", str(precision))
    print("Recall:", str(recall))
