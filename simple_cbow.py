import numpy as np 
import torch 
import torch.optim as optim
from torch import nn 
from torch.autograd import Variable
import read_data 
import pickle 

class CBOW(nn.Module):


    def __init__(self, vocab_size, embedding_dim, output_dim,pretrained_embedding, pretrain=True):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.linear = nn.Linear(embedding_dim, output_dim)
        if pretrain:
            self.embeddings.weight = nn.Parameter(pretrained_embedding)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        bow = torch.sum(embeds, 0)
        logits = self.linear(bow)     
        return logits

# Reading Glove based on https://github.com/A-Jacobson/CNN_Sentence_Classification/blob/master/WordVectors.ipynb 
glove_path = 'glove/glove.6B.50d.txt'
def load_glove(path):
    with open(path) as f:
        glove = {} 
        for line in f.readlines():
            values = line.split() 
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove[word] = vector 
        return glove 

def load_glove_embeddings(path, word2idx, embedding_dim=50):
    with open(path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float()

glove = load_glove(glove_path)

# # READ IN IMDB VOCABULARY 
# vocab_file = open('aclImdb/imdb.vocab', 'r')
# vocab = vocab_file.read().split('\n')
# word2idx = {word: idx for idx, word in enumerate(vocab)}
# print(word2idx.keys())
w2i_file = open("dl4nlt/data/w2i.pkl", "rb")
word2idx = pickle.load(w2i_file)
embedding_glove = load_glove_embeddings(glove_path, word2idx)
# embedding_glove
# # print(embedding_glove.size())
# # emb = nn.Embedding(embedding.size(0), embedding.size(1), padding_idx=0)
# # emb.weight = nn.Parameter(embedding)
# TO DO FIX PADDING TOKEN 
model = CBOW(embedding_glove.size(0), embedding_glove.size(1), 2, embedding_glove)
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), 2e-3)

train_neg = read_data.read_data("aclImdb/train/neg")
y_neg = [0] * len(train_neg)
train_pos = read_data.read_data("aclImdb/train/pos")
y_pos = [1] * len(train_pos)

train_data = train_neg + train_pos
y_data = y_neg + y_pos 
# print(train_data[0])
del train_neg
del train_pos

imdb_dataset = read_data.ImdbDataset(train_data, y_data)
dataloader = read_data.DataLoader(imdb_dataset, batch_size=10)
train_file = open("dl4nlt/data/train_data.pkl", "rb")
train_labels = open("dl4nlt/data/train_labels.pkl", "rb")
x = pickle.load(train_file)
y = pickle.load(train_labels)

imdb_dataset = read_data.ImdbDataset(x, y)
dataloader = read_data.DataLoader(imdb_dataset, batch_size=10, collate_fn=read_data.collate)

for idx, data in enumerate(dataloader):
    if idx == 1:
        break
    x, y = data
    model(x.long())






