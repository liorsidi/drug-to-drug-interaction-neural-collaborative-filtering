"""Train and test LSTM classifier"""
from keras.backend import binary_crossentropy
from keras.callbacks import EarlyStopping, History, TensorBoard
from keras.utils import to_categorical
from pandas import Series, DataFrame
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tldextract import tldextract

import dga_classifier.data as data
import numpy as np
import pickle
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.cross_validation import train_test_split
from tensorflow.python.client import device_lib
from keras.utils import np_utils


from keras.layers import Input, Dense,  Lambda, K
from keras.models import Model


def load_simpleAE_model(path):
    """
    the function load a pickle file for constructing the LSTM_DGA and an h5 file for loading the lstm model (architecture and weights)
    :param path: the model name path - example: /models/all_data/model_type1
    :return: a LSTM_DGA class with an lstm model
    """

    print 'loading model from ' + path
    model_dict = pickle.load(open(path + ".p", 'rb') )
    model = SimpleDomainEmbedder(**model_dict)
    model.model = load_model(path + ".h5")
    return model

import numpy as np


class SimpleDomainEmbedder(object):

    def __init__(self,max_epoch = 25,batch_size = 256, embedding_dim=64,dropout=0.5,
                 vocabulary = 1434,optimizer = 'adadelta', is_CBOW = True, window = 5
                 ,  **kwargs):
        super(SimpleDomainEmbedder, self).__init__()

        self.is_CBOW = is_CBOW
        self.loss = 'categorical_crossentropy'
        if is_CBOW:
            self.loss = 'categorical_crossentropy'

        self.batch_size = batch_size

        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.window = window
        self.vocabulary = vocabulary

        self.model = None
        self.encoder = None

    def __str__(self):
        s = 'w2v_'
        if self.is_CBOW:
            s += 'cbow'
        else:
            s += 'skipgram'
        return s

    def save(self, path):
        params = dict()
        primitive = (int, str, dict, list, float)
        for k, v in self.__dict__.items():
            if isinstance(v, primitive):
                params[k] = v

        print params
        pickle.dump(params, open(path + self.__str__() + '.p', "wb"))
        char_embbeding_model_file = path + self.__str__() + ".h5"
        self.model.save(char_embbeding_model_file)
        if self.encoder != None:
            char_embbeding_encoder_file = path + self.__str__() + "_encoder.h5"
            self.encoder.save(char_embbeding_encoder_file)

        self.model.save(path + str(self) + '.h5')
        return path + str(self)

    def build_model(self):

        """Build LSTM model"""
        from numpy.random import seed
        seed(1)
        from tensorflow import set_random_seed
        set_random_seed(2)

        # cbow = Sequential()
        # cbow.add(Embedding(input_dim=self.vocabulary, output_dim=self.embedding_dim, input_length=self.window * 2))
        # cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(self.embedding_dim,)))
        # cbow.add(Dense(self.vocabulary, activation='softmax'))
        # cbow.compile(loss=self.loss, optimizer=self.optimizer)

        cbow = Input(shape=(self.window * 2,self.vocabulary,))
        cbow = Embedding(input_dim=self.vocabulary, output_dim=self.embedding_dim, input_length=self.window * 2)(cbow)
        cbow = Lambda(lambda x: K.mean(x, axis=1), output_shape=(self.embedding_dim,))(cbow)
        self.encoder = cbow
        cbow = Dense(self.vocabulary, activation='softmax')(cbow)
        cbow.compile(loss=self.loss, optimizer=self.optimizer)

        cbow.summary()
        self.model = cbow


    def fit(self, X, y):
        # print GPU or CPU
        print 'fitting lstm model...' + str(self)
        print(device_lib.list_local_devices())

        self.model.fit(X, y,
                       batch_size=self.batch_size,
                       epochs=self.max_epoch,
                       shuffle=True,
                       verbose=1,
                       callbacks= [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')])

    def predict(self, X):
        self.model.predict(X)

    def get_embedding(self, X):
        self.encoder.predict(X)



