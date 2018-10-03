import torch

from os import listdir
from os.path import isfile, join

import sys
sys.path.append('../processing/')
from get_data import get_dataset
from lstm_gather import LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Hyperparameters aanpassen voor eigen model!!!
BATCH_SIZE = 64
NUM_CLASSES = 2
EMBEDDING_DIM = 50
SEQUENCE_LENGTH = 500
NUM_HIDDEN = 256
NUM_LAYERS = 1


def get_files(folder):
    """
    Returns all model files in a folder.
    """

    return [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]


def get_model(file_name):
    """
    Loads the model and optimizer.
    """

    model = LSTM(EMBEDDING_DIM, BATCH_SIZE, NUM_CLASSES, NUM_HIDDEN,
    NUM_LAYERS).to(device)

    saved = torch.load(file_name, map_location="cpu")
    model.load_state_dict(saved["state_dict"])

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

            outputs = model(x, doc_lengths)
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

    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)
    return precision, recall


if __name__ == "__main__":
    val_loader = get_dataset("validation", BATCH_SIZE)
    # test_loader = get_dataset("test", BATCH_SIZE)

    file_names = get_files("model_states")

    for file_name in file_names:
        model = get_model(file_name)
        evaluation = evaluate(model, val_loader)
        print(file_name, str(evaluation))
        break
