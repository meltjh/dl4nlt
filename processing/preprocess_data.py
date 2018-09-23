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
    embedded_data = []
    # Needed for padding later in model
    document_lengths = []
    review_id = []
    
    all_document_lengths = []

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
        
        all_document_lengths.append(review_length)
        
        if review_length <= MAX_REVIEW_LENGTH:
            review_id.append(file_name)
            indices = seq2idx(splitted_review, w2i)
            embedded_sentence = idx2embed(indices, embeddings)
            embedded_data.append(embedded_sentence)
            document_lengths.append(review_length)
    
    return embedded_data, document_lengths, review_id, all_document_lengths

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

def plot_frequencies(dataset_type, freq_neg, freq_pos):
    # Make a combined frequencies list
    freq_combined = []
    freq_combined.extend(freq_neg)
    freq_combined.extend(freq_pos)
    
    # Sort the counter of the frequencies in order to make the labels be the indices of the x axis
    sorted_x = sorted(Counter(freq_combined).items(), key=lambda pair: pair[0], reverse=False)
    labels, values = zip(*sorted_x)
    indexes = labels
    
    fig, ax1 = plt.subplots()
    
    # Plot histogram
    ax1.set_xlabel('Amount of words')
    ax1.set_ylabel('Amount of documents (bars)')
    ax1.bar(indexes, values, alpha=0.3)
    
    # Plot the lines
    ax2 = ax1.twinx()
    ax2.set_ylabel('Amount of documents in dataset given the word limit (lines)')
    ax2.grid(True)
    ax2.set_ylim(0, sum(values)*1.05) # The limit have to get set in order to make the y=0 be equal for both axis. The * >1 is for y top margin
    for freqs, label in [(freq_neg, "Negatives"), (freq_pos, "Positive"), (freq_combined, "Combined")]:
        # Sort the counter of the frequencies in order to make the labels be the indices of the x axis
        sorted_x = sorted(Counter(freqs).items(), key=lambda pair: pair[0], reverse=False)
        labels, values = zip(*sorted_x)
        indexes = labels
        
        # For each index sum the previous values
        y_total = []
        tot_val = 0
        for value in values:
            tot_val += value
            y_total.append(tot_val)
            
        ax2.plot(indexes, y_total, label=label, alpha=0.6)
        
    fig.legend(loc=4, bbox_to_anchor=(0.88, 0.1))
    ax1.set_title("{} dataset".format(dataset_type.capitalize()))
    plt.tight_layout()
    plt.show()

def save_dataset(w2i, embeddings, dataset_type):
    """
    Read out the imdb data and get the word ids and the word embeddings. 
    Word embeddings: sentences = [[word_embed1, word_embed2, word_embed3], [word_embed4, word_embed5, word_embed6]]
    Labels are also saved.
    """
    # Get the positive and negative datasets.
    print("-- Retrieving datasets from folders")
    dataset_neg_embedded, doc_lengths_neg, doc_ids_neg, freq_neg = preprocess(IMDB_PATH + "/{}/neg".format(dataset_type), w2i, embeddings)
    dataset_pos_embedded, doc_lengths_pos, doc_ids_pos, freq_pos = preprocess(IMDB_PATH + "/{}/pos".format(dataset_type), w2i, embeddings)
    
    # Plot the frequencies
#    plot_frequencies(dataset_type, freq_neg, freq_pos)
    
    if dataset_type == "train":
        dataset_neg = (dataset_neg_embedded, doc_lengths_neg, doc_ids_neg)
        dataset_pos = (dataset_pos_embedded, doc_lengths_pos, doc_ids_pos)
        dataset_neg, dataset_pos = create_equal_datasets(dataset_neg, dataset_pos)

        splitted_dataset_neg = split_dataset(dataset_neg[0], dataset_neg[1], dataset_neg[2])
        splitted_dataset_pos = split_dataset(dataset_pos[0], dataset_pos[1], dataset_pos[2])
                
        training_word_embedded = splitted_dataset_neg[0] + splitted_dataset_pos[0]
        training_labels = [0]*len(splitted_dataset_neg[0]) + [1]*len(splitted_dataset_pos[0])
        training_doc_lengths = splitted_dataset_neg[1] + splitted_dataset_pos[1]
        training_doc_ids = splitted_dataset_neg[2] + splitted_dataset_pos[2]
        save_single_dataset("train", training_word_embedded, training_labels, training_doc_lengths, training_doc_ids)
        
        validation_word_embedded = splitted_dataset_neg[3] + splitted_dataset_pos[3]
        validation_labels = [0]*len(splitted_dataset_neg[3]) + [1]*len(splitted_dataset_pos[3])
        validation_doc_lengths = splitted_dataset_neg[4] + splitted_dataset_pos[4]
        validation_doc_ids = splitted_dataset_neg[5] + splitted_dataset_pos[5]
        save_single_dataset("validation", validation_word_embedded, validation_labels, validation_doc_lengths, validation_doc_ids)

    else:
        test_word_embedded = dataset_neg_embedded + dataset_pos_embedded
        test_labels = [0]*len(dataset_neg_embedded) + [1]*len(dataset_neg_embedded)
        test_doc_lengths = doc_lengths_neg + doc_lengths_pos
        test_doc_ids = doc_ids_neg + doc_ids_pos

        save_single_dataset("test", test_word_embedded, test_labels, test_doc_lengths, test_doc_ids)

def create_equal_datasets(dataset_neg, dataset_pos):
    num_neg = len(dataset_neg[0])
    num_pos = len(dataset_pos[0])
    if num_neg < num_pos:
        dataset_pos = cutoff_dataset(dataset_pos, num_neg)
    else:
        dataset_neg = cutoff_dataset(dataset_neg, num_pos)
    return dataset_neg, dataset_pos
        
def cutoff_dataset(dataset, max_samples):
    word_embedded = dataset[0][:max_samples]
    doc_lengths = dataset[1][:max_samples]
    doc_ids = dataset[2][:max_samples]
    return (word_embedded, doc_lengths, doc_ids)
    
def split_dataset(word_embedded, doc_lengths, doc_ids):
    num_samples = len(doc_lengths)
    
    zipped = list(zip(word_embedded, doc_lengths, doc_ids))
    random.shuffle(zipped)
    shuffled_word_embedded, shuffled_doc_lengths, shuffled_doc_ids = (zip(*zipped))
  
    training_size = round(num_samples*TRAINING_PERCENTAGE)
    
    training_word_embedded = shuffled_word_embedded[:training_size]
    training_doc_lengths = shuffled_doc_lengths[:training_size]
    training_doc_ids = shuffled_doc_ids[:training_size]

    validation_word_embedded = shuffled_word_embedded[training_size:]
    validation_doc_lengths = shuffled_doc_lengths[training_size:]
    validation_doc_ids = shuffled_doc_ids[training_size:]
    
    print("Training size", len(training_word_embedded))
    print("Validation size", len(validation_word_embedded))
    print("Total size", len(doc_lengths))
    
    return (training_word_embedded, training_doc_lengths, training_doc_ids, 
            validation_word_embedded, validation_doc_lengths, validation_doc_ids)
    
def save_single_dataset(dataset_type, embedded, labels, doc_lengths, doc_ids):
    # Concatenate to get one big set.
    save_pickle(embedded, "{}/{}/data_embedded.pkl".format(DATA_SAVE_PATH, dataset_type))
    del embedded
    
    save_pickle(labels, "{}/{}/labels.pkl".format(DATA_SAVE_PATH, dataset_type))
    del labels
    
    save_pickle(doc_lengths, "{}/{}/doc_lengths.pkl".format(DATA_SAVE_PATH, dataset_type))
    del doc_lengths
    
    save_pickle(doc_ids, "{}/{}/doc_ids.pkl".format(DATA_SAVE_PATH, dataset_type))
    del doc_ids

def save_pickle(data, filename):
    """
    Save the data as a .pkl file.
    """
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    f.close()

if __name__ == "__main__":
    save_all_datasets()
    