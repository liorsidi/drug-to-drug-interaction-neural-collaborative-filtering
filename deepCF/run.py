from __future__ import division

import os

import itertools
from keras.callbacks import EarlyStopping
from scipy.stats import spearmanr
import random
import pandas as pd
import time

from deepCF import NCF


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
    else:
        node1 = list(data['node1'])
        node2 = list(data['node2'])
        y = None
    return node1, node2, y


def main(args):

    model_params = {
        'dropout': [0.2,0.4],
        'layers': [2,4],
        'factors': [16,32],
        'deep': [True],
        'batch_size': [512],
        'class_weight': [{0: 1., 1: 12.}],
        'kernel_regularizer' : [0.001,0.0001],
        'activation': ['tanh'],
        'callback': [[EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=0, mode='auto')]],
        'epochs': [50],

    }

    model_params = {
        'dropout': [0.4],
        'layers': [2],
        'factors': [16],
        'deep': [True],
        'batch_size': [512],
        'class_weight': [{0: 1., 1: 12.}],
        'kernel_regularizer' : [0.001],
        'activation': ['tanh'],
        'callback': [[EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=0, mode='auto')]],
        'epochs': [25],

    }

    home_path = '/home/ise/Desktop/hackathon-bgu_data_hack/'
    train_full_node1, train_full_node2, train_full_y = create_data( home_path + 'data_for_kaggle/full_train.csv', 'train')
    test_full_node1, test_full_node2, _ = create_data(home_path + 'data_for_kaggle/kaggle_test.csv', 'eval')

    train_node1, train_node2, train_y = create_data(home_path + 'data_for_testing/eval_train.csv', 'train')
    test_node1, test_node2, test_y = create_data(home_path + 'data_for_testing/eval_test.csv', 'test')

    models = generate_entity(NCF,model_params)

    for model in models:
        print model.__str__()
        model.fit(train_node1,train_node2,train_y)
        preds1 = model.predict(test_node1,test_node2)
        preds2 = model.predict(test_node2,test_node1)

        res = {}
        res['preds1'] = preds1
        res['preds2'] = preds2
        res['preds'] = (preds2 + preds1)/2.
        res['label'] = test_y
        res['node1'] = test_node1
        res['node2'] = test_node2
        res = pd.DataFrame(res).sort_values('preds', ascending = False)
        res.to_csv(home_path + 'deepCF/DCF_test.csv', columns = ['node1','node2','preds','label'], index=False)
        print average_precision_at_k(100, list(res['label']))

    # for model in models:
    #     model.fit(train_full_node1, train_full_node2, train_full_y)
    #     preds1 = model.predict(test_full_node1, test_full_node2)
    #     preds2 = model.predict(test_full_node2, test_full_node1)
    #     print (np.mean(preds2 - preds1))
    #
    #     res = {}
    #     res['node1'] = test_full_node1
    #     res['node2'] = test_full_node2
    #     res['preds1'] = preds1
    #     res['preds2'] = preds2
    #     res['preds_mean'] = (preds2 + preds1) / 2.
    #     #res['label'] = test_y
    #
    #     res = pd.DataFrame(res).sort_values('preds_mean', ascending=False)
    #     res["node1_node2"] = res["node1"].map(str) + "_"+ res["node2"].map(str)
    #     res.to_csv(home_path + 'deepCF/DCF_for_kaggle.csv', columns = ['node1_node2'], index=False)

import sys
main(sys.argv[1:])