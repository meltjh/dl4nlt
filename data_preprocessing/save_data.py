from nltk.tokenize import word_tokenize
from os import listdir
from os.path import isfile, join
import pickle


def read_data(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    data = []

    for file_name in files:
        text_file = open(path + "/" + file_name, 'r')
        review = text_file.read().lower()
        data.append(word_tokenize(review))
    return data


def save_pickle(data, filename):
    pickle_file = open(filename, "wb")
    pickle.dump(train_data, pickle_file)
    pickle_file.close()


if __name__ == "__main__":
    train_neg = read_data("aclImdb/train/neg")
    train_pos = read_data("aclImdb/train/pos")
    train_data = train_neg + train_pos

    train_labels = [0]*len(train_neg) + [1]*len(train_pos)

    save_pickle(train_data, "train_labels.pkl")
    save_pickle(train_labels, "train_labels.pkl")

    del train_neg
    del train_pos
    del train_labels


    test_neg = read_data("aclImdb/test/neg")
    test_pos = read_data("aclImdb/test/pos")
    test_data = test_neg + test_pos

    test_labels = [0]*len(test_neg) + [1]*len(test_pos)

    save_pickle(test_data, "test_data.pkl")
    save_pickle(test_labels, "test_labels.pkl")
