import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense, dot,Input
from keras.models import Sequential, load_model
from keras.preprocessing.text import one_hot
from keras.regularizers import l2
from keras.utils import to_categorical
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import pickle

def load_simpleAE_model(path):
    """
    the function load a pickle file for constructing the LSTM_DGA and an h5 file for loading the lstm model (architecture and weights)
    :param path: the model name path - example: /models/all_data/model_type1
    :return: a LSTM_DGA class with an lstm model
    """

    print 'loading model from ' + path
    items_encoder = pickle.load(open(path + 'item_encoder.p', "wb"))
    model_dict = pickle.load(open(path + ".p", 'rb'))
    model = NCF(**model_dict)
    model.items_encoder = items_encoder
    model.model = load_model(path + ".h5")
    return model

class CFModel(Sequential):
    def __init__(self, n_users, m_items,optimizer='adadelta', item2item = True,
                 loss='mean_squared_error',activation = 'tanh', factors=20, deep=True,
                 layers=2, dropout=0.5,kernel_regularizer = 0.01, **kwargs):
        """
        a NCF keras implementation that support generic architecture design

        """
        P = Sequential()
        Q = Sequential()
        if factors == 0:  # relative
            user_factors = int(m_items / 10)
            items_factors = int(m_items / 10)
        else:
            user_factors = factors
            items_factors = factors
        if deep:
            # if item2item:
            #     emb_user = Embedding(n_users, user_factors, input_length=1)
            #     emb_item = emb_user
            # else:
            #     emb_user = Embedding(n_users, user_factors, input_length=1)
            #     emb_item = Embedding(m_items, items_factors, input_length=1)
            # P.add(emb_user)
            # P.add(Reshape((user_factors,)))
            # Q.add(emb_item)
            # Q.add(Reshape((items_factors,)))

            P.add(Embedding(n_users, user_factors, input_length=1))
            P.add(Reshape((user_factors,)))
            Q.add(Embedding(m_items, items_factors, input_length=1))
            Q.add(Reshape((items_factors,)))
            super(CFModel, self).__init__(**kwargs)
            if layers == 0:  # regular SVD
                self.add(Merge([P, Q], mode='dot', dot_axes=1))
            else:  # deeper framework
                self.add(Merge([P, Q], mode='concat'))
                for l in range(layers):
                    self.add(Dropout(dropout))
                    self.add(Dense(user_factors,kernel_regularizer=l2(kernel_regularizer),
                                   activation=activation))
                self.add(Dropout(dropout))
                self.add(Dense(1))#, activation='linear'))
        else:  # wide
            P.add(Dense(n_users,input_dim =1))
            Q.add(Dense(m_items,input_dim=1))
            super(CFModel, self).__init__(**kwargs)
            self.add(Merge([P, Q], mode='concat', dot_axes=1))
            self.add(Dense(1, activation=activation))

        self.compile(loss=loss, optimizer=optimizer)

class NCF(object):
    def __init__(self, epochs=10,
                 batch_size=1024,callback=None,shuffle=True,class_weight = {0: 1., 1: 10.},
                 optimizer='adadelta',
                 loss='binary_crossentropy', activation='tanh', factors=20, deep=True,activation_hidden = 'tanh',
                 layers=2, dropout=0.5,kernel_regularizer = 0.01,):
        """
        a NCF implementation for question sequencing
        :param s_id: student id
        :param S: all students answers
        :param questionaire: questionaires
        """
        self.model = CFModel(1, 1)
        self.factors = factors
        if self.factors == 0:  # relative
            factors = int(self.n_items / 10)
        self.deep = deep
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.callback = callback
        self.shuffle = shuffle
        self.le_users = None
        self.loss = loss
        self.optimizer = optimizer
        self.class_weight = class_weight
        self.kernel_regularizer = kernel_regularizer
        self.activation_hidden = activation_hidden
        self.n_items = 0

    def predict(self,item1, item2):
        item1_encoded = self.items_encoder.transform(item1)
        item2_encoded = self.items_encoder.transform(item2)
        preds = self.model.predict([item1_encoded, item2_encoded],batch_size=2048).flatten()
        return preds

    def __str__(self):
        s = 'deepCF'
        str_values = ['factors','deep','layers','dropout','epochs','activation','loss','class_weight','kernel_regularizer',
                      'activation_hidden']
        for k in str_values:
            s = s + "_" + str(self.__dict__[k])
        return s

    def build_model(self):
        input1 = Input(shape=(1,))
        input2 = Input(shape=(1,))
        emb_item = Embedding(self.n_items, self.factors, input_length=1)
        item1 = emb_item(input1)
        item1 = Reshape((self.factors,))(item1)
        item2 = emb_item(input2)
        item2 = Reshape((self.factors,))(item2)

        concat = dot([item1, item2], axes = 1)

        # concat = Merge([item1, item2], mode='concat')

        for layer in range(self.layers):
            concat = Dropout(self.dropout)(concat)
            concat = Dense(self.factors, activation=self.activation_hidden)(concat)

        output_model = Dense(1, activation=self.activation)(concat)

        model = Model(inputs=[input1,input2], outputs=output_model)

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def save(self, path):
        params = dict()
        primitive = (int, str, dict, list, float)
        for k, v in self.__dict__.items():
            if isinstance(v, primitive):
                params[k] = v
        print params
        pickle.dump(params, open(path + self.__str__() + '.p', "wb"))
        pickle.dump(self.items_encoder, open(path + self.__str__() + 'item_encoder.p', "wb"))
        model_file = path + self.__str__() + ".h5"
        self.model.save(model_file)

        return path + str(self)

    def fit(self, item1, item2, y):
        self.items_encoder = LabelEncoder()

        n_items = len(set(item1 + item2))
        self.items_encoder.fit(list(set(item1 + item2)))

        item1_encoded = self.items_encoder.transform(item1)
        item2_encoded = self.items_encoder.transform(item2)

        # self.model = CFModel(n_items, n_items,optimizer=self.optimizer,
        #          loss=self.loss,activation = self.activation, factors=self.factors,
        #                      deep=self.deep, layers=self.layers, dropout=self.dropout,kernel_regularizer= self.kernel_regularizer)
        #
        self.model = self.build_model()

        self.model.summary()
        self.model.fit([item1_encoded, item2_encoded], y, batch_size=self.batch_size, epochs=self.epochs,
                       shuffle=self.shuffle,class_weight=self.class_weight, verbose=2)

