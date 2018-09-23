from nltk.tokenize import word_tokenize
from os import listdir
from os.path import isfile, join
import pickle
import string
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt

IMDB_PATH = "aclImdb"
GLOVE_PATH = "glove/glove.6B.50d.txt"
DATA_SAVE_PATH = "../data"
MAX_REVIEW_LENGTH = 500
PADDING_KEY = "PAD"
TRAINING_PERCENTAGE = 0.8

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
    reviews = []
    
    REV_LENGTH = []
    j = 0

    for file_name in files:
        text_file = open(path + "/" + file_name, 'r')
        # Convert the review to lowercase
        review = text_file.read().lower()
        # Tokenize the review
        tokenized_review = str(word_tokenize(review))
        # Remove punctuation from the review
        stripped_review = tokenized_review.translate(translator)
        splitted_review = stripped_review.split()
        review_length = len(splitted_review)
        
        REV_LENGTH.append(review_length)
        
        if review_length <= MAX_REVIEW_LENGTH:
            j += 1
            reviews.append(splitted_review)
            indices = seq2idx(splitted_review, w2i)
            embedded_sentence = idx2embed(indices, embeddings)
            idx_data.append(indices)
            embedded_data.append(embedded_sentence)
            document_lengths.append(review_length)
    
    return idx_data, embedded_data, document_lengths, reviews, REV_LENGTH

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
            indices.append(w2i[PADDING_KEY])
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
    print("Loading imdb vocabulary")
    # Get word2id mapping
    vocab_file = open(IMDB_PATH + "/imdb.vocab", "r")
    vocab = vocab_file.read().split("\n")
    w2i = {word: idx for idx, word in enumerate(vocab)}
    i2w = {idx: word for idx, word in enumerate(vocab)}

    # Last id is for padding
    padding_id = len(vocab)
    w2i["PAD"] = padding_id
    i2w[padding_id] = PADDING_KEY
    
    # Get the embeddings
    embeddings = load_glove_embeddings(w2i)
    
    # Save per dataset (training/test)
    print("Saving training data")
    save_dataset(w2i, embeddings, "train")  
    print("Saving test data")
    save_dataset(w2i, embeddings, "test")
    
    save_pickle(w2i, "{}/w2i.pkl".format(DATA_SAVE_PATH))
    save_pickle(i2w, "{}/i2w.pkl".format(DATA_SAVE_PATH))
    
    print("Finished")
    
def load_glove_embeddings(word2idx, embedding_dim=50):
    """
    Returns the Glove embeddings of the words that occur in the IMDB dataset.
    """
    with open(GLOVE_PATH) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
        return embeddings

def save_dataset(w2i, embeddings, dataset_type):
    """
    Read out the imdb data and get the word ids and the word embeddings. 
    Save the data in two ways:
        1. word ids: sentences = [[idx1, idx2, idx3], [idx4, idx5, idx6]]
        2. word embeddings: sentences = [[word_embed1, word_embed2, word_embed3], [word_embed4, word_embed5, word_embed6]]
    Labels are also saved.
    """
    # Get the positive and negative datasets.
    print("-- Retrieving datasets from folders")
    dataset_neg_idx, dataset_neg_embedded, doc_lengths_neg, reviews_neg, freq_neg = preprocess(IMDB_PATH + "/{}/neg".format(dataset_type), w2i, embeddings)
    dataset_pos_idx, dataset_pos_embedded, doc_lengths_pos, reviews_pos, freq_pos = preprocess(IMDB_PATH + "/{}/pos".format(dataset_type), w2i, embeddings)
    
    if dataset_type == "train":
        freq_neg.extend(freq_pos)
            
        counter = Counter(freq_neg)
        
        labels, values = zip(*counter.items())
        indexes = np.arange(len(labels))
        width = 1
        plt.bar(indexes, values, width)
        plt.show()
        
        ekwjdhewkjdhw
        
        
        splitted_dataset_neg = split_dataset(dataset_neg_idx, dataset_neg_embedded, doc_lengths_neg, reviews_neg)
        splitted_dataset_pos = split_dataset(dataset_pos_idx, dataset_pos_embedded, doc_lengths_pos, reviews_pos)
        
        training_word_idx = splitted_dataset_neg[0] + splitted_dataset_pos[0]
        training_word_embedded = splitted_dataset_neg[1] + splitted_dataset_pos[1]
        training_labels = [0]*len(splitted_dataset_neg[0]) + [1]*len(splitted_dataset_pos[0])
        training_doc_lengths = splitted_dataset_neg[2] + splitted_dataset_pos[2]
        training_reviews = splitted_dataset_neg[3] + splitted_dataset_pos[3]
        
        save_single_dataset("train", training_word_idx, training_word_embedded, training_labels, training_doc_lengths, training_reviews)
        
        validation_word_idx = splitted_dataset_neg[4] + splitted_dataset_pos[4]
        validation_word_embedded = splitted_dataset_neg[5] + splitted_dataset_pos[5]
        training_labels = [0]*len(splitted_dataset_neg[4]) + [1]*len(splitted_dataset_pos[4])
        validation_doc_lengths = splitted_dataset_neg[6] + splitted_dataset_pos[6]
        validation_reviews = splitted_dataset_neg[7] + splitted_dataset_pos[7]
        
        save_single_dataset("validation", validation_word_idx, validation_word_embedded, training_labels, validation_doc_lengths, validation_reviews)

    else:
        test_word_idx = dataset_neg_idx + dataset_pos_idx
        test_word_embedded = dataset_neg_embedded + dataset_pos_embedded
        test_labels = [0]*len(dataset_neg_idx) + [1]*len(dataset_pos_idx)
        test_doc_lengths = doc_lengths_neg + doc_lengths_pos
        test_reviews = reviews_neg + reviews_pos

        save_single_dataset("test", test_word_idx, test_word_embedded, test_labels, test_doc_lengths, test_reviews)
        
def split_dataset(word_idx, word_embedded, doc_lengths, reviews):
    num_samples = len(word_idx)
    
    zipped = list(zip(word_idx, word_embedded, doc_lengths, reviews))
    random.shuffle(zipped)
    word_idx, word_embedded, doc_lengths, reviews = (zip(*zipped))
  
    training_size = round(num_samples*TRAINING_PERCENTAGE)
    
    training_word_idx = word_idx[:training_size]
    training_word_embedded = word_embedded[:training_size]
    training_doc_lengths = doc_lengths[:training_size]
    training_shuffled_reviews = reviews[:training_size]
    
    validation_word_idx = word_idx[training_size:]
    validation_word_embedded = word_embedded[training_size:]
    validation_doc_lengths = doc_lengths[training_size:]
    validation_shuffled_reviews = reviews[training_size:]
    
    print("Training size", len(training_word_idx))
    print("Validation size", len(validation_word_idx))
    print("Total size", len(word_idx))
    
    return (training_word_idx, training_word_embedded, training_doc_lengths, training_shuffled_reviews,
            validation_word_idx, validation_word_embedded, validation_doc_lengths, validation_shuffled_reviews)
    
def save_single_dataset(dataset_type, idx, embedded, labels, doc_lengths, reviews):
    # Concatenate to get one big set.
    print("-- Dataset ids")
    save_pickle(idx, "{}/{}/data_idx.pkl".format(DATA_SAVE_PATH, dataset_type))
    del idx

    print("-- Dataset embedded")
    save_pickle(embedded, "{}/{}/data_embedded.pkl".format(DATA_SAVE_PATH, dataset_type))
    del embedded
    
    print("-- Labels")
    save_pickle(labels, "{}/{}/labels.pkl".format(DATA_SAVE_PATH, dataset_type))
    del labels
    
    print("-- Lengths")
    save_pickle(doc_lengths, "{}/{}/doc_lengths.pkl".format(DATA_SAVE_PATH, dataset_type))
    del doc_lengths
    
    print("-- Lengths")
    save_pickle(reviews, "{}/{}/reviews.pkl".format(DATA_SAVE_PATH, dataset_type))
    del reviews

def save_pickle(data, filename):
    """
    Save the data as a .pkl file.
    """
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    f.close()

if __name__ == "__main__":
    save_all_datasets()
    