import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import matplotlib.pyplot as plt


# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"

BATCH_SIZE = 64
N_EPOCHS = 20
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001

# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    embds = np.zeros(embedding_dim)
    for word in sent.text:
        embds += word_to_vec.get(word, np.zeros_like(embds))
    return embds / len(sent.text)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    vec = np.zeros((size,))
    vec[ind] = 1
    return vec


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    size = len(word_to_ind)
    vec = np.zeros(size, )
    for word in sent.text:
        vec += get_one_hot(size, word_to_ind[word])
    return vec / len(sent.text)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    maps = {}
    for i, w in enumerate(words_list):
        maps[w] = i
    return maps


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    vecs = []
    for i in range(len(sent.text)):
        if i == seq_len: break
        word = sent.text[i]
        if word in word_to_vec: vecs.append(
            word_to_vec.get(word, np.zeros(embedding_dim)))
    for j in range(len(vecs) - seq_len): vecs.append(np.zeros(embedding_dim))
    return np.array(vecs)


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager:
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True,
                 dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path,
                                                               split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[
                TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {
                "word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(
                                         words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {
                "word_to_vec": create_or_load_slim_w2v(words_list),
                "embedding_dim": embedding_dim
            }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {
            k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs)
            for
            k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN) for
                                k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array(
            [sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers,
                                  bidirectional=True, batch_first=True,
                                  dropout=dropout)

        self.linear = torch.nn.Linear(hidden_dim * 2, 1)
        self.sig = torch.nn.Sigmoid()

    def forward(self, text):
        out_lstm, zz = self.lstm(text)
        o = out_lstm[:, -1, :]
        return self.linear(o)

    def predict(self, text):
        return self.sig(self.forward(text))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, 1)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        out = self.linear(x)
        return self.sig(out)


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    accurate_predictions = 0
    for i in range(len(preds)):
        if preds[i] == y[i]:
            accurate_predictions += 1
    return accurate_predictions / len(preds)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    total_loss = 0
    total_accuracy = 0
    my_device = get_available_device()
    for x, y in data_iterator:
        local_x = x.to(my_device)
        local_y = y.to(my_device)
        optimizer.zero_grad()
        pred = model(local_x.type(torch.FloatTensor)).transpose(0, 1).squeeze(0)
        curr_loss = criterion(pred, local_y)
        total_loss += curr_loss
        curr_loss.backward()
        optimizer.step()
        total_accuracy += binary_accuracy(torch.round(nn.Sigmoid()(pred)), y)

    data_set_size = len(data_iterator)
    accuracy = total_accuracy / data_set_size
    loss = total_loss / data_set_size
    return accuracy, loss


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    total_loss = 0
    total_accuracy = 0
    my_device = get_available_device()
    for x, y in data_iterator:
        local_x = x.to(my_device)
        local_y = y.to(my_device)
        pred = model(local_x.type(torch.FloatTensor)).transpose(0, 1).squeeze(
            0)
        curr_loss = criterion(pred, local_y)
        total_loss += curr_loss
        curr_loss.backward()
        total_accuracy += binary_accuracy(torch.round(nn.Sigmoid()(pred)), y)
    data_set_size = len(data_iterator)
    accuracy = total_accuracy / data_set_size
    loss = total_loss / data_set_size

    return accuracy, loss


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    cor_count = 0
    batch_size = data_iter.batch_size
    iter_to_epoch = int(len(data_iter.dataset) // batch_size)

    for j in range(iter_to_epoch):
        for i, (X, y) in enumerate(
                data_iter[j * batch_size:(j + 1) * batch_size]):
            lbls = y.reshape(data_iter.batch_size, 1).astype(torch.FloatTensor)
            pred = model.predict(X.astype(torch.FloatTensor))
            cor_count += int(binary_accuracy(pred, lbls) * len(lbls))

    return 100 * cor_count / int(len(data_iter.dataset))


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """

    criterion = nn.BCEWithLogitsLoss()
    train_iter = data_manager.get_torch_iterator(TRAIN)
    validation_iter = data_manager.get_torch_iterator(VAL)

    optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    train_loss = []
    train_accuracy = []
    validation_loss = []
    validation_accuracy = []

    for e in range(n_epochs):
        print(f"training epoch {e}/{n_epochs}:")
        loss, accuracy = train_epoch(model, train_iter, optimizer, criterion)
        train_loss.append(loss)
        train_accuracy.append(accuracy)

        loss, accuracy = evaluate(model, validation_iter, criterion)
        validation_loss.append(loss)
        validation_accuracy.append(accuracy)

    return {"loss": [train_loss, validation_loss], "accuracy": [train_accuracy, validation_accuracy]}


def test_and_show_results(train_results, data_manager, n_epochs, model_name):
    epochs_vals = range(n_epochs)
    for data_type, values in train_results.items():
        plt.plot(epochs_vals, values[0], label="train loss")
        plt.plot(epochs_vals, values[1], label="validation loss")
        plt.title(f'{model_name} - {data_type} over training/validation')
        plt.xlabel('Epochs')
        plt.ylabel(data_type)
        plt.legend()
        plt.show()
        plt.savefig(f"output_files/{model_name}_{data_type}")

    test_iter = data_manager.get_torch_iterator(TEST)



def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manager = DataManager(ONEHOT_AVERAGE, batch_size=BATCH_SIZE)
    embedding_dim = data_manager.get_input_shape()[0]
    model = LogLinear(embedding_dim)
    train_results = train_model(model, data_manager, n_epochs=20, lr=0.01,
                                weight_decay=0.0001)
    save_pickle(train_results, "pickle_files/one_hot")
    test_and_show_results(train_results, data_manager, 20, "LogLinearOneHot")



def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manager = DataManager(W2V_AVERAGE, batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    embedding_dim = data_manager.get_input_shape()[0]
    model = LogLinear(embedding_dim)
    train_results = train_model(model, data_manager, n_epochs=20, lr=0.01,
                                weight_decay=0.0001)

    test_and_show_results(train_results, data_manager, 20,
                          "LogLinearWord2Vec")


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(W2V_SEQUENCE, batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(embedding_dim=W2V_EMBEDDING_DIM, hidden_dim=100, n_layers=1, dropout=.5)
    train_results = train_model(model, data_manager, n_epochs=4, lr=0.001,
                                weight_decay=0.0001)
    test_and_show_results(train_results, data_manager, 4,
                          "LSTMWord2Vec")


if __name__ == '__main__':
    train_log_linear_with_one_hot()
    # train_log_linear_with_w2v()
    # train_lstm_with_w2v()
