import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb

def create_ngram_set(input_list, ngram_value=2):
    """
    Create a set of n-grams
    :param input_list: [1, 2, 3, 4, 9]
    :param ngram_value: 2
    :return: {(1, 2),(2, 3),(3, 4),(4, 9)}
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list by appending n-grams values
    :param sequences:
    :param token_indice:
    :param ngram_range:
    :return:
    """
    new_seq = []
    for input in sequences:
        new_list = input[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_seq.append(new_list)
    return new_seq

ngram_range = 2
max_features = 20000
max_len = 400
batch_size = 32
embedding_dims = 50
epochs = 5

print('loading data...')
x_train, y_train, x_test, y_test = imdb.load_data(num_words=max_features)
train_mean_len = np.mean(list(map(len, x_train)), dtype=int)
test_mean_len = np.mean(list(map(len, x_test)), dtype=int)
print(len(x_train), 'train seq')
print(len(x_test), 'test seq')
print('Average train sequence length: {}'.format(train_mean_len))
print('Average test sequence length: {}'.format(test_mean_len))