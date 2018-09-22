from nltk.tokenize import word_tokenize
from os import listdir
from os.path import isfile, join
import pickle
import string
import numpy as np

IMDB_PATH = "aclImdb"
GLOVE_PATH = "glove/glove.6B.50d.txt"
DATA_SAVE_PATH = "../data"
MAX_REVIEW_LENGTH = 500

def preprocess(path, w2i, embeddings):
    """
    Finds all reviews in a folder and converts them to lowercase, tokenizes
    them and removes punctuation. If the length of the review is longer than
    MAX_REVIEW_LENGTH words, don't use it. Returns a list of lists with indices of the words
    that occur in the review.
    """
    files = [f for f in listdir(path) if isfile(join(path, f))]
    translator = str.maketrans('', '', string.punctuation)
    idx_data = []
    embedded_data = []
    # Needed for padding later in model
    document_lengths = []

    for file_name in files:
        text_file = open(path + "/" + file_name, 'r')
        # Convert the review to lowercase
        review = text_file.read().lower()
        # Tokenize the review
        tokenized_review = str(word_tokenize(review))
        # Remove punctuation from the review
        review = tokenized_review.translate(translator)
        review_length = len(tokenized_review)
        if review_length <= MAX_REVIEW_LENGTH:
            splitted_review = review.split()
            indices = seq2idx(splitted_review, w2i)
            embedded_sentence = idx2embed(indices, embeddings)
            idx_data.append(indices)
            embedded_data.append(embedded_sentence)
            document_lengths.append(review_length)
    return idx_data, embedded_data, document_lengths

def seq2idx(sequence, w2i):
    """
    Convert a sequence of words to a sequence of their indices in the w2i
    dictionary.
    """
    indices = []

    for word in sequence:
        if word in w2i:
            indices.append(w2i[word])
        else:
            indices.append(w2i["UNK"])
    return indices

def idx2embed(indices, embeddings):
    """
    Convert a sequence of word ids to a sequence of their Glove embeddings.
    """
    sentence_length = len(indices)
    embedded_sentence = np.zeros((sentence_length, embeddings.shape[1]))
    for i in range(sentence_length):
        idx = indices[i]
        word_embedding = embeddings[idx]
        embedded_sentence[i] = word_embedding
    return embedded_sentence

def save_all_datasets():
    """
    Save the training and test data.
    """
    # Get word2id mapping
    vocab_file = open(IMDB_PATH + "/imdb.vocab", "r")
    vocab = vocab_file.read().split("\n")
    w2i = {word: idx for idx, word in enumerate(vocab)}
    i2w = {idx: word for idx, word in enumerate(vocab)}

    # Last id is for padding
    padding_id = len(vocab)
    w2i["PAD"] = padding_id
    i2w[padding_id] = "PAD"
    
    # Get the embeddings
    embeddings = load_glove_embeddings(GLOVE_PATH, w2i)
    
    # Save per dataset (training/test)
    print("Saving training data")
    save_single_dataset(IMDB_PATH, w2i, embeddings, "train")  
    print("Saving test data")
    save_single_dataset(IMDB_PATH, w2i, embeddings, "test")
    
    save_pickle(w2i, "{}/w2i.pkl".format(DATA_SAVE_PATH))
    save_pickle(i2w, "{}/i2w.pkl".format(DATA_SAVE_PATH))
    
    print("Finished")
    
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

def save_single_dataset(w2i, embeddings, dataset_type):
    """
    Read out the imdb data and get the word ids and the word embeddings. 
    Save the data in two ways:
        1. word ids: sentences = [[idx1, idx2, idx3], [idx4, idx5, idx6]]
        2. word embeddings: sentences = [[word_embed1, word_embed2, word_embed3], [word_embed4, word_embed5, word_embed6]]
    Labels are also saved.
    """
    # Get the positive and negative datasets.
    print("-- Retrieving datasets from folders")
    dataset_neg_idx, dataset_neg_embedded, doc_lengths_neg = preprocess(IMDB_PATH + "/{}/neg".format(dataset_type), w2i, embeddings)
    dataset_pos_idx, dataset_pos_embedded, doc_lengths_pos = preprocess(IMDB_PATH + "/{}/pos".format(dataset_type), w2i, embeddings)
    
    # Concatenate to get one big set.
    print("-- Dataset id's")
    dataset_idx = dataset_neg_idx + dataset_pos_idx
    save_pickle(dataset_idx, "{}/{}/data_idx.pkl".format(DATA_SAVE_PATH, dataset_type))
    del dataset_idx

    print("-- Dataset embedded")
    dataset_embedded = dataset_neg_embedded + dataset_pos_embedded
    save_pickle(dataset_embedded, "{}/{}/data_embedded.pkl".format(DATA_SAVE_PATH, dataset_type))
    del dataset_embedded, dataset_neg_embedded, dataset_pos_embedded
    
    print("-- Labels")
    dataset_labels = [0]*len(dataset_neg_idx) + [1]*len(dataset_pos_idx)
    save_pickle(dataset_labels, "{}/{}/labels.pkl".format(DATA_SAVE_PATH, dataset_type))
    del dataset_labels, dataset_neg_idx, dataset_pos_idx
    
    print("-- Lengths")
    doc_lengths = doc_lengths_neg + doc_lengths_pos
    save_pickle(doc_lengths, "{}/{}/doc_lengths.pkl".format(DATA_SAVE_PATH, dataset_type))
    del doc_lengths, doc_lengths_neg, doc_lengths_pos
    
def save_pickle(data, filename):
    """
    Save the data as a .pkl file.
    """
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    f.close()

if __name__ == "_main_":
    save_all_datasets()