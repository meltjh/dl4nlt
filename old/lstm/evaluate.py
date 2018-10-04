import torch

import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import re
import pandas as pd

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

    files = [join(folder, f) for f in listdir(folder) if
    isfile(join(folder, f))]

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(files, key=alphanum_key)


def get_model(file_name):
    """
    Loads the model and optimizer.
    """

    model = LSTM(EMBEDDING_DIM, BATCH_SIZE, NUM_CLASSES, NUM_HIDDEN,
    NUM_LAYERS).to(device)

    saved = torch.load(file_name, map_location="cpu")
    model.load_state_dict(saved["state_dict"])

    return model

def get_doc_ids(doc_ids,indices):
    docs = [doc_ids[ind].numpy().tolist() for ind in indices]
    list_str = [str(doc[0][0]) + '_' + str(doc[0][1]) + '.txt' for doc in docs]
    return list_str

def create_error_tuple(doc_ids, fp_ind, fn_ind):
    fp_doc = get_doc_ids(doc_ids, fp_ind)
    fn_doc = get_doc_ids(doc_ids, fn_ind)

    return fp_doc, fn_doc


def evaluate(model, dataloader):
    """
    Evaluates the model by calculating the accuracy, precision and recall.
    """

    num_batches = len(dataloader)
    sum_accuracy = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    error_vals_fp = []
    error_vals_fn = [] 

    with torch.no_grad():
        for batch_i, data in enumerate(dataloader):
            x, y, doc_ids, doc_lengths = data
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).long().to(device)

            outputs = model(x, doc_lengths)
            outputs = torch.max(outputs, 1)[1]

            accuracy = get_accuracy(outputs, y)
            sum_accuracy += accuracy.item()

            tp, fp, fn, fp_ind, fn_ind = get_TP_FP_FN(outputs, y)
            fp_doc, fn_doc = create_error_tuple(doc_ids, fp_ind, fn_ind)
            error_vals_fp.extend(fp_doc)
            error_vals_fn.extend(fn_doc)
            true_positives += tp
            false_positives += fp
            false_negatives += fn
    dict_val = {'false_postive':error_vals_fp, 'false_negative':error_vals_fn}
    df = pd.DataFrame.from_dict(dict_val, orient='index')
    #### CHANGE FILE NAME TO CSV ACCORDINGLY ####
    df.to_csv('256error')
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

    false_pos_ind = (torch.where((targets==0) & (predictions==1), true, false)).nonzero()
    false_neg_ind = (torch.where((targets==1) & (predictions==0), true, false)).nonzero()


    true_positives = torch.sum(torch.where((targets==1) & (predictions==1),
                                           true, false)).item()
    false_positives = torch.sum(torch.where((targets==0) & (predictions==1),
                                            true, false)).item()
    false_negatives = torch.sum(torch.where((targets==1) & (predictions==0),
                                            true, false)).item()
    return true_positives, false_positives, false_negatives, false_pos_ind, false_neg_ind


def get_precision_recall(true_positives, false_positives, false_negatives):
    """
    Calculates the precision and recall.
    """

    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)
    return precision, recall


def get_training_results(file_name):
    """
    Retrieves the accuracies and losses that were obtained from a file.
    """

    accuracies = []
    losses = []

    with open(file_name, "r") as f:
        for line in f.readlines():
            split_line = line.split(",")
            accuracy = float(split_line[2].split("=")[1])
            accuracies.append(accuracy)
            loss = float(split_line[3].split("=")[1])
            losses.append(loss)
    return accuracies, losses


def plot_results(train_loss, train_accuracy, validation_accuracy, file_name):
    """
    Plots the training loss, training accuracy and validation accuracy.
    """

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy %', color='C0')
    ax1.tick_params('y', colors='C0')

    ax1.plot(train_accuracy, color='C0', linestyle=":", label="Train")
    ax1.plot(validation_accuracy, color='C0', linestyle='--',
             label="Validation")

    ax1.legend()

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='C1')
    ax2.tick_params('y', colors='C1')

    ax2.plot(train_loss, color='C1', linestyle="-", label="Loss")

    fig.tight_layout()
    plt.savefig(file_name + ".png")
    plt.show()
    # plt.savefig(file_name + ".png")


def save_results(file_names, accuracies):
    validation_file = file_names[-1].split("/")[1]
    validation_file = validation_file[:validation_file.rfind("_")]
    with open('val_' + validation_file + '.txt', "w") as f:
        for accuracy in accuracies:
            f.write(str(accuracy) + "\n")
    f.close()
    return validation_file 


if __name__ == "__main__":
    ### For validation set and plots
    # val_accuracy = []
    # val_loader = get_dataset("validation", BATCH_SIZE)
    # file_names = get_files("model_states")

    # for file_name in file_names:
    #     model = get_model(file_name)
    #     accuracy, _, _ = evaluate(model, val_loader)
    #     print(file_name, str(accuracy))
    #     val_accuracy.append(accuracy)

    # file_name = save_results(file_names, val_accuracy)
    # train_accuracy, train_loss = get_training_results("train_result_lr1-e5-256.txt")
    # plot_results(train_loss, train_accuracy, val_accuracy, file_name)

    # For final evalution on test set
    test_loader = get_dataset("test", BATCH_SIZE)
    # file_name = "model_states/checkpoint19.pth"
    file_name = "model1e-3/lr0.001_batchsize64_embeddim50_hidden256_layers1_checkpoint31.pth"
    model = get_model(file_name)
    accuracy, precision, recall = evaluate(model, test_loader)
    print("File:", file_name)
    print("Accuracy:", str(accuracy))
    print("Precision:", str(precision))
    print("Recall:", str(recall))
