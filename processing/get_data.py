from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np 
import pickle
from preprocess_data import load_glove_embeddings

class ImdbDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y 

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x,y

def collate(batch):
    batch_size = len(batch)
    x = torch.ones((batch_size, 500))
    y = torch.ones((batch_size))
    for i in range(batch_size):
        batch_n = cut_off(batch[i][0], 500)
        x[i] = torch.from_numpy(np.array(batch_n))
        y[i] = torch.from_numpy(np.array(batch[i][1]))
    return x,y 

def cut_off(vector, cut_val):
    if len(vector) > cut_val:
        vector = vector[:cut_val]
    else:
        padding_len = cut_val - len(vector)
        vector = vector + (padding_len * [89527])
    return vector

def get_datasets():
    """ Returns the stored train, validation and test set. """
    train_file = open("../data/train_data.pkl", "rb")
    train_labels = open("../data/train_labels.pkl", "rb")
    x = pickle.load(train_file)
    y = pickle.load(train_labels)
    
    imdb_dataset = ImdbDataset(x, y)
    dataloader = DataLoader(imdb_dataset, batch_size=10, collate_fn=collate)
    
    return dataloader