from os import listdir
from os.path import isfile, join
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np 
import pickle
from load_glove import load_glove_embeddings

class ImdbDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y 

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # print(self.x[idx])
        x = self.x[idx]
        y = self.y[idx]
        return x,y

def read_data(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    data = []

    for file_name in files[:10]:
        text_file = open(path + "/" + file_name, 'r')
        review = text_file.read().lower()
        # str_token = ' '.join(word_tokenize(review))
        # data.append(str_token)
        data.append(word_tokenize(review))
    return data

def collate(batch):
    batch_size = len(batch)
    # x = []
    # y = []
    x = torch.ones((batch_size, 500))
    y = torch.ones((batch_size))
    for i in range(batch_size):
        batch_n = cut_off(batch[i][0], 500)
        x[i] = torch.from_numpy(np.array(batch_n))
        y[i] = torch.from_numpy(np.array(batch[i][1]))
    return x,y 

def cut_off(vector, cut_val):
    ''' '''
    if len(vector) > cut_val:
        vector = vector[:cut_val]
    else:
        padding_len = cut_val - len(vector)
        vector = vector + (padding_len * [89527])
    return vector




# train_neg = read_data("aclImdb/train/neg")
# y_neg = [0] * len(train_neg)
# train_pos = read_data("aclImdb/train/pos")
# y_pos = [1] * len(train_pos)

# train_data = train_neg + train_pos
# y_data = y_neg + y_pos 
# # print(train_data[0])
# del train_neg
# del train_pos

# imdb_dataset = ImdbDataset(train_data, y_data)
# dataloader = DataLoader(imdb_dataset, batch_size=10, collate_fn=collate)

# for idx, data in enumerate(dataloader):
#     if idx == 1:
#         break
#     x, y = data 

    # print('target', y)


def get_data():
    '''Gekopieerd van simple_cbow.'''
    train_file = open("../data/train_data.pkl", "rb")
    train_labels = open("../data/train_labels.pkl", "rb")
    x = pickle.load(train_file)
    y = pickle.load(train_labels)
    
    imdb_dataset = ImdbDataset(x, y)
    dataloader = DataLoader(imdb_dataset, batch_size=10, collate_fn=collate)
    
    return dataloader