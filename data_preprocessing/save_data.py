from nltk.tokenize import word_tokenize
from os import listdir
from os.path import isfile, join
import pickle
import string


def read_data(path, w2i):
    """
    Finds all reviews in a folder and converts them to lowercase, tokenizes
    them and removes punctuation. If the length of the review is longer than
    500 words, don't use it. Returns a list of lists with indices of the words
    that occur in the review.
    """

    files = [f for f in listdir(path) if isfile(join(path, f))]
    translator = str.maketrans('', '', string.punctuation)
    data = []

    for file_name in files:
        text_file = open(path + "/" + file_name, 'r')
        # Convert the review to lowercase
        review = text_file.read().lower()
        # Tokenize the review
        tokenized_review = str(word_tokenize(review))
        # Remove punctuation from the review
        review = tokenized_review.translate(translator)

        if len(tokenized_review) <= 500:
            indices = seq2idx(review.split(), w2i)
            data.append(indices)
    return data


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


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    f.close()


if __name__ == "__main__":
    train_file = open("train_data.pkl", "rb")
    train_data = pickle.load(train_file)

    print(train_data)

    # vocab_file = open("aclImdb/imdb.vocab", "r")
    # vocab = vocab_file.read().split("\n")
    # w2i = {word: idx for idx, word in enumerate(vocab)}
    # w2i["UNK"] = len(vocab)
    #
    #
    # train_neg = read_data("aclImdb/train/neg", w2i)
    # train_pos = read_data("aclImdb/train/pos", w2i)
    # train_data = train_neg + train_pos
    #
    # train_labels = [0]*len(train_neg) + [1]*len(train_pos)
    #
    # save_pickle(train_data, "train_data.pkl")
    # save_pickle(train_labels, "train_labels.pkl")
    #
    # del train_neg
    # del train_pos
    # del train_labels
    #
    #
    # test_neg = read_data("aclImdb/test/neg", w2i)
    # test_pos = read_data("aclImdb/test/pos", w2i)
    # test_data = test_neg + test_pos
    #
    # test_labels = [0]*len(test_neg) + [1]*len(test_pos)
    #
    # save_pickle(test_data, "test_data.pkl")
    # save_pickle(test_labels, "test_labels.pkl")
