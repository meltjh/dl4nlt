from nltk.tokenize import word_tokenize
from os import listdir
from os.path import isfile, join
import pickle
import string
from load_glove import load_glove_embeddings

def read_data(path, w2i, embeddings):
    """
    Finds all reviews in a folder and converts them to lowercase, tokenizes
    them and removes punctuation. If the length of the review is longer than
    500 words, don't use it. Returns a list of lists with indices of the words
    that occur in the review.
    """

    files = [f for f in listdir(path) if isfile(join(path, f))]
    translator = str.maketrans('', '', string.punctuation)
    idx_data = []
    embedded_data = []

    for file_name in files:
        text_file = open(path + "/" + file_name, 'r')
        # Convert the review to lowercase
        review = text_file.read().lower()
        # Tokenize the review
        tokenized_review = str(word_tokenize(review))
        # Remove punctuation from the review
        review = tokenized_review.translate(translator)

        if len(tokenized_review) <= 500:
            splitted_review = review.split()
            indices = seq2idx(splitted_review, w2i)
            embedded_sentence = idx2embed(indices, embeddings)
            idx_data.append(indices)
            embedded_data.append(embedded_sentence)
    return idx_data, embedded_data

def save_dataset(imdb_folder, w2i, embeddings, dataset_type, data_save_folder):
    """
    Read out the imdb data and get the word ids and the word embeddings. 
    Save the data in two ways:
        1. word ids: sentences = [[idx1, idx2, idx3], [idx4, idx5, idx6]]
        2. word embeddings: sentences = [[word_embed1, word_embed2, word_embed3], [word_embed4, word_embed5, word_embed6]]
    Labels are also saved.
    """
    # Get the positive and negative datasets.
    dataset_neg_idx, dataset_neg_embedded = read_data(imdb_folder + "/{}/neg".format(dataset_type), w2i, embeddings)
    dataset_pos_idx, dataset_pos_embedded = read_data(imdb_folder + "/{}/pos".format(dataset_type), w2i, embeddings)
    
    # Concatenate to get one big set.
    dataset_idx = dataset_neg_idx + dataset_pos_idx
    dataset_embedded = dataset_neg_embedded + dataset_pos_embedded
    dataset_labels = [0]*len(dataset_neg_idx) + [1]*len(dataset_pos_idx)
    
    # Save the data.
    save_pickle(dataset_idx, "{}/{}_data_idx.pkl".format(data_save_folder, dataset_type))
    save_pickle(dataset_embedded, "{}/{}_data_embedded.pkl".format(data_save_folder, dataset_type))
    save_pickle(dataset_labels, "{}/{}_labels.pkl".format(data_save_folder, dataset_type))

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
    embedded_sentence = []
    for idx in indices:
        word_embedding = embeddings[idx]
        embedded_sentence.append(word_embedding)
    return embedded_sentence

def save_pickle(data, filename):
    """
    Save the data as a .pkl file.
    """
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    f.close()
    
def save_data():
    """
    Save the training and test data.
    """
    # Get word2id mapping
    imdb_folder = "aclImdb"
    vocab_file = open(imdb_folder + "/imdb.vocab", "r")
    vocab = vocab_file.read().split("\n")
    w2i = {word: idx for idx, word in enumerate(vocab)}
    w2i["UNK"] = len(vocab)
    
    # Get the embeddings
    glove_path = 'glove/glove.6B.50d.txt'
    embeddings = load_glove_embeddings(glove_path, w2i)
    data_folder = "../data"
    
    # Save per dataset (training/test)
    save_dataset(imdb_folder, w2i, embeddings, "train", data_folder)    
    save_dataset(imdb_folder, w2i, embeddings, "test", data_folder)
    

if __name__ == "__main__":
    save_data()
#    train_file = open("../data/train_data.pkl", "rb")
#    train_data = pickle.load(train_file)
#
#    print(train_data)
