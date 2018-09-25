# Returns the saved datasets in batches. Use the function get_datasets() to 
# get the train, validation and test set as a Dataloader.
# By looping through a set you get a batch with:
# - the embedded reviews (batch size, maximum review length, embedding dimension)
# - the labels (1, batch size)
# - the document ids (batchsize, 2), one original document id looks like "1_5.txt", 
#   this is changed to [1, 5].
# - the document lengths (1, batch_size), needed to know how much is padded.

import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import pickle
from preprocess_data import MAX_REVIEW_LENGTH, DATA_SAVE_PATH

#DATA_SAVE_PATH = "data"

class ImdbDataset(Dataset):
    """ Uses the embedded review, label (0 or 1), document id and document length. """
    def __init__(self, x, y, doc_ids, doc_lengths):
        self.x = x
        self.y = y
        self.doc_ids = doc_ids
        self.doc_lengths = doc_lengths

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        y = self.y[i]
        # Doc ids are also stored for analysis
        doc_id = self.doc_ids[i]
        doc_length = self.doc_lengths[i]
        return {"x": x, "y": y, "doc_id": doc_id, "doc_length": doc_length}

def collate(batch):
    """ Returns one batch with the embedded review, labels and 
    doc ids (as [1,5] instead of "1_5.txt"). """
    embedding_dim = batch[0]["x"].shape[1]
    batch_size = len(batch)
    x = torch.zeros((batch_size, MAX_REVIEW_LENGTH, embedding_dim))
    y = torch.zeros((batch_size))
    doc_ids = torch.zeros((batch_size, 2)) # doc_ids will always consist of 2 numbers.
    doc_lengths = torch.zeros((batch_size))
    for i in range(batch_size):
        batch_n = add_padding(batch[i]["x"], MAX_REVIEW_LENGTH)
        x[i] = torch.from_numpy(batch_n)
        y[i] = torch.from_numpy(np.array(batch[i]["y"]))
        doc_ids[i] = torch.from_numpy(np.array(strip_doc_id(batch[i]["doc_id"])))
        doc_lengths[i] = torch.from_numpy(np.array(batch[i]["doc_length"]))
    return x, y, doc_ids, doc_lengths

def strip_doc_id(doc_id):
    """ PyTorch Dataloader cannot handle strings, so the doc_ids are stripped.
    E.g. if the original was 1_5.txt, this now becomes [1, 5] """
    ext_removed = doc_id.rsplit(".txt")[0]
    doc_id_list = [int(num) for num in ext_removed.split("_")]
    return doc_id_list

def add_padding(matrix, cut_val):
    """ Adds zero matrix if the review length is shorter than the maximum 
    length. """
    num_words, embedding_dim = matrix.shape
    if num_words < MAX_REVIEW_LENGTH:
        padding_len = MAX_REVIEW_LENGTH - num_words
        padding = np.zeros((padding_len, embedding_dim))
        matrix = np.vstack([matrix, padding])
    return matrix

def get_single_dataset(dataset_type, batch_size):
    """ Read the saved files and transform this into a Dataloader. """
    embedded_file = open("{}/{}/data_embedded.pkl".format(DATA_SAVE_PATH, dataset_type), "rb")
    labels = open("{}/{}/labels.pkl".format(DATA_SAVE_PATH, dataset_type), "rb")
    doc_ids = open("{}/{}/doc_ids.pkl".format(DATA_SAVE_PATH, dataset_type), "rb")
    doc_lengths = open("{}/{}/doc_lengths.pkl".format(DATA_SAVE_PATH, dataset_type), "rb")
    
    x = pickle.load(embedded_file)
    y = pickle.load(labels)
    ids = pickle.load(doc_ids)
    lengths = pickle.load(doc_lengths)
    
    imdb_dataset = ImdbDataset(x, y, ids, lengths)
    dataloader = DataLoader(imdb_dataset, batch_size, collate_fn=collate, shuffle=True)
    return dataloader

def get_dataset(dataset, batch_size=10):
    """ Returns the stored train, validation and test set. """
    if dataset == "train":
        print("Getting train")
        dataset = get_single_dataset("train", batch_size)
    elif dataset == "test":
        print("Getting validation")
        dataset = get_single_dataset("validation", batch_size)
    else:
        print("Getting test")
        dataset = get_single_dataset("test", batch_size)
    
    return dataset
