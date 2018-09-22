import numpy as np
import torch

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
    """
    Returns the Glove embeddings of the words that occur in the IMDB dataset.
    """
    with open(path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
        return embeddings