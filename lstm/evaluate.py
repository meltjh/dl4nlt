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


def get_accuracy(predictions, targets):
    """
    Calculates the accuracy.
    """

    _, pred = torch.max(predictions, 1)
    num_correct = torch.sum(pred == targets, dtype=torch.float, dim = 0)
    accuracy = num_correct / pred.shape[0] * 100
    return accuracy


def evaluate(model, dataloader):
    num_batches = len(dataloader)
    sum_accuracy = 0

    with torch.no_grad():
        for batch_i, data in enumerate(dataloader):
            x, y, doc_ids, doc_lengths = data

            x = torch.tensor(x).to(device)
            y = torch.tensor(y).long().to(device)

            outputs = model(x, doc_lengths)
            accuracy = get_accuracy(outputs, y)
            sum_accuracy += accuracy.item()

    return sum_accuracy/num_batches

if __name__ == "__main__":
    val_loader = get_dataset("validation", BATCH_SIZE)
    # test_loader = get_dataset("test", BATCH_SIZE)

    file_names = get_files("model_states")

    for file_name in file_names:
        model = get_model(file_name)
        accuracy = evaluate(model, val_loader)
        print(file_name, str(accuracy))
