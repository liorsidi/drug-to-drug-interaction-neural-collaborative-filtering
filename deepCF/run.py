from __future__ import division

import os

import itertools
from keras.callbacks import EarlyStopping
from scipy.stats import spearmanr
import random
import pandas as pd
import time

from sklearn import preprocessing

from deepCF import NCF
from deepCF_net import NCF_net


def get_index_product(params):
    i = 0
    params_index = {}
    for k, v in params.items():
        params_index[k] = i
        i += 1
    params_list = [None] * len(params_index.values())
    for name, loc in params_index.items():
        params_list[loc] = params[name]

    params_product = list(itertools.product(*params_list))
    params_product_dicts = []
    for params_value in params_product:
        params_dict = {}
        for param_name, param_index in params_index.items():
            params_dict[param_name] = params_value[param_index]
        params_product_dicts.append(params_dict)

    return params_product_dicts


def generate_entity(model_class, model_params):
    """
    generate all possible combination of the class with the parmeters
    :param model_class:
    :param model_params:
    :return:
    """
    models = []
    model_params_product = get_index_product(model_params)
    for model_param in model_params_product:
        models.append(model_class(**model_param))
    return models

def average_precision_at_k(k, class_correct):
    # return average precision at k.
    # more examples: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    # and: https://www.kaggle.com/c/avito-prohibited-content#evaluation
    # class_correct is a list with the binary correct label ordered by confidence level.
    assert k <= len(class_correct) and k > 0
    score = 0.0
    hits = 0.0
    for i in range(k):
        if class_correct[i] == 1:
            hits += 1.0
        score += hits / (i + 1.0)
    score /= k
    return score

import numpy as np


def create_data(csv_path,type):
    data = pd.read_csv(csv_path)
    if type == 'train':
        node1 = list(data['node1']) + list(data['node2'])
        node2 = list(data['node2']) + list(data['node1'])
        y = list(data['label']) + list(data['label'])
    elif type == 'test':
        node1 = list(data['node1'])
        node2 = list(data['node2'])
        y = list(data['label'])
    elif type == 'net_train':
        y = list(data['label'])
        node1 = list(data['node1'])
        node2 = list(data['node2'])
        p = data[['p2', 'p3']].values
        min_max_scaler = preprocessing.MinMaxScaler()
        p_v = min_max_scaler.fit_transform(p)

        return node1, node2, p_v, y
    elif type == 'net_test':
        node1 = list(data['node1'])
        node2 = list(data['node2'])
        data = data.convert_objects(convert_numeric=True)
        # data.f
        #data['p4'] = data['p4'].astype('float32')
        p = data[[ 'p2', 'p3']].values
        min_max_scaler = preprocessing.MinMaxScaler()
        p_v = min_max_scaler.fit_transform(p)
        return node1, node2, p_v, None
    else:
        node1 = list(data['node1'])
        node2 = list(data['node2'])
        y = None
    return node1, node2, y


def main(args):

    model_params = {
        'dropout': [0.4],
        'layers': [2],
        'factors': [16],
        'deep': [True],
        'batch_size': [2048],
        'class_weight': [{0: 1., 1: 12.}],
        'kernel_regularizer' : [0.001],
        'activation': ['tanh'],
        'callback': [[EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=0, mode='auto')]],
        'epochs': [3],
    }

    model_params = {
        'dropout': [0.2,0.4],
        'layers': [2,4],
        'factors': [16,32],
        'deep': [True],
        'batch_size': [1024],
        'class_weight': [{0: 1., 1: 12.},{0: 1., 1: 6.}],
        'kernel_regularizer' : [0.001,0.0001],
        'activation': ['tanh','relu'],
        'callback': [[EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=0, mode='auto')]],
        'epochs': [3],
    }

    model_params = {
        'dropout': [0.25],
        'layers': [2],
        'factors': [64],
        'deep': [True],
        'batch_size': [1024],
        'class_weight': [{0: 1., 1: 12.}],
        'kernel_regularizer': [0.001],
        'activation': ['tanh'],
        'callback': [[EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=0, mode='auto')]],
        'epochs': [50],
        'seed': [5,32],
    }

    home_path = '/home/ise/Desktop/hackathon-bgu_data_hack/'
    train_full_node1, train_full_node2, train_full_y = create_data( home_path + 'data_for_kaggle/full_train.csv', 'train')
    test_full_node1, test_full_node2, _ = create_data(home_path + 'data_for_kaggle/kaggle_test.csv', 'eval')

    train_node1, train_node2, train_y = create_data(home_path + 'data_for_testing/eval_test.csv', 'train')
    test_node1, test_node2, test_y = create_data(home_path + 'data_for_testing/eval_train.csv', 'test')

    # model_params = {
    #     'dropout': [0.25],
    #     'layers': [2],
    #     'factors': [64],
    #     'deep': [True],
    #     'batch_size': [1024],
    #     'class_weight': [{0: 1., 1: 12.}],
    #     'kernel_regularizer': [0.0001],
    #     'activation': ['tanh'],
    #     'callback': [[EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=0, mode='auto')]],
    #     'epochs': [5],
    #     'seed': [5,32],
    # }
    # models = generate_entity(NCF, model_params)
    # models_res = []
    # for model in models:
    #     print model.__str__()
    #     model.fit(train_node1,train_node2,train_y)
    #     preds1 = model.predict(test_node1,test_node2)
    #     preds2 = model.predict(test_node2,test_node1)
    #
    #     res = {}
    #     res['preds1'] = preds1
    #     res['preds2'] = preds2
    #     res['preds'] = (preds2 + preds1)/2.
    #     res['label'] = test_y
    #     res['node1'] = test_node1
    #     res['node2'] = test_node2
    #     res = pd.DataFrame(res).sort_values('preds', ascending = False)
    #     res.to_csv(home_path + 'deepCF/DCF_test.csv', columns = ['node1','node2','preds','label'], index=False)
    #     prec =  average_precision_at_k(100, list(res['label']))
    #     print prec
    #
    #     model_res = model.__dict__
    #     model_res['kprec'] = prec
    #     models_res.append(model_res)
    # pd.DataFrame(models_res).to_csv(home_path + 'deepCF/models_res.csv')

    model_params = {
        'dropout': [0.2],
        'layers': [2],
        'factors': [32],
        'optimizer' : ['adam'],
        'deep': [True],
        'batch_size': [2048],
        'class_weight': [{0: 1., 1: 5.}],
        'kernel_regularizer': [0.0001],
        'activation': ['tanh'],
        'activation_hidden' : ['tanh'],
        'epochs': [100],
        'seed': [32],
    }
    models = generate_entity(NCF, model_params)



    model_params = {
        'dropout': [0.1],
        'layers': [3],
        'prev' : [False],
        'post': [True],
        'factors': [12],
        'reduce_dim' : [8],
        'optimizer' : ['adam'],
        'deep': [True],
        'batch_size': [2048],
        'class_weight': [{0: 1., 1: 4.}],
        'kernel_regularizer': [0.0001],
        'activation': ['tanh'],
        'activation_hidden' : ['tanh'],
        'epochs': [20],
        'seed': [32],
    }
    # models = models + generate_entity(NCF, model_params)
    models =  generate_entity(NCF, model_params)
    for model in models:
        model.fit(train_full_node1, train_full_node2, train_full_y,False)
        preds1 = model.predict(test_full_node1, test_full_node2)
        preds2 = model.predict(test_full_node2, test_full_node1)
        print (np.mean(preds2 - preds1))

        res = {}
        res['node1'] = test_full_node1
        res['node2'] = test_full_node2
        res['preds1'] = preds1
        res['preds2'] = preds2
        res['preds_mean'] = (preds2 + preds1) / 2.
        #res['label'] = test_y

        res = pd.DataFrame(res).sort_values('preds_mean', ascending=False)
        res["node1_node2"] = res["node1"].map(str) + "_"+ res["node2"].map(str)
        res.to_csv(home_path + 'deepCF/' + model.__str__()+ 'DCF_for_kaggle.csv', columns = ['node1','node2','preds_mean'], index=False)


def main2(args):

    home_path = '/home/ise/Desktop/hackathon-bgu_data_hack/'
    train_node1, train_node2, train_p, train_y = create_data(home_path + 'data_for_kaggle/all_features_2c1ee3c0_combined.csv', 'net_train')
    test_node1, test_node2, test_p, _ = create_data(home_path + 'data_for_kaggle/all_features_71a2e5cf.csv', 'net_test')

    model_params = {
        'dropout': [0.1],
        'layers': [2],
        'prev' : [False],
        'post': [True],
        'factors': [32],
        'reduce_dim' : [12],
        'optimizer' : ['adam'],
        'deep': [True],
        'batch_size': [2048],
        'class_weight': [{0: 1., 1: 4.}],
        'kernel_regularizer': [0.0001],
        'activation': ['tanh'],
        'activation_hidden' : ['tanh'],
        'epochs': [20],
        'seed': [32],
    }

    models =  generate_entity(NCF_net, model_params)
    for model in models:
        model.fit(train_node1, train_node2, train_p,train_y,True)
        preds1 = model.predict(test_node1, test_node2, test_p)
        preds2 = model.predict(test_node2, test_node1, test_p)
        print (np.mean(preds2 - preds1))

        res = {}
        res['node1'] = test_node1
        res['node2'] = test_node2
        res['preds1'] = preds1
        res['preds2'] = preds2
        res['preds_mean'] = (preds2 + preds1) / 2.
        #res['label'] = test_y

        res = pd.DataFrame(res).sort_values('preds_mean', ascending=False)
        res["node1_node2"] = res["node1"].map(str) + "_"+ res["node2"].map(str)
        res.to_csv(home_path + 'deepCF/' + model.__str__()+ 'DCF_net_for_kaggle.csv', columns = ['node1','node2','preds_mean'], index=False)


import sys

def fix_data_before_submission(path = './data_for_kaggle/kaggle_test.csv'):

    # my_data = genfromtxt('./data_for_kaggle/kaggle_test.csv', delimiter=',')

    # np.apply_along_axis(myfunction, axis=1, arr=mymatrix)



    df = pd.read_csv(path)

    df_out = pd.DataFrame()



    df_out['node1'] = df['node1'].where(df.node1 < df.node2, df['node2'])

    df_out['node2'] = df['node2'].where(df.node1 < df.node2, df['node1'])

    df_out["node1_node2"] = df_out["node1"].map(str) + "_"+ df_out["node2"].map(str)

    df_out.to_csv('/home/ise/Desktop/hackathon-bgu_data_hack/deepCF/DCF_net2_deep_dig_for_kaggle.csv', columns = ['node1_node2'], index=False)





    print "done first part"

    for idx, row in df_out.iterrows():

        if row[0] > row[1]:

            print "OH SHIT"

file = "/home/ise/Desktop/hackathon-bgu_data_hack/deepCF/deepCF_12_True_3_0.1_20_tanh_binary_crossentropy_{0: 1.0, 1: 4.0}_0.0001_tanh_32_False_True_8DCF_for_kaggle.csv"
# file = "/home/ise/Desktop/hackathon-bgu_data_hack/deepCF/deepCF_24_True_3_0.4_20_sigmoid_binary_crossentropy_{0: 1.0, 1: 4.0}_0.0001_tanh_32_False_True_12DCF_net_for_kaggle.csv"
fix_data_before_submission(file)
main(sys.argv[1:])
# main2(sys.argv[1:])
