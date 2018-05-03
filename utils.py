import random
import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.model_selection import train_test_split



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



def get_train():
    kfold_num = 10
    graph = nx.read_edgelist('new_train_train_no_header.csv', delimiter=',', nodetype=str)
    non_edges = nx.non_edges(graph)
    df = pd.DataFrame(list(non_edges))
    df['label'] = 0
    df = df.rename(columns={0: 'node1', 1: 'node2'})
    df2 = pd.read_csv('new_train_train.csv')
    df2['label'] = 1
    full_df = df2.append(df)
    print "\ttrain:\n\tnum of ones: {}\n\tnum of zeros:{}".format(df2.shape[0],df.shape[0])
    full_df.to_csv('eval_test.csv', index=False)

def get_test():
    kfold_num = 10
    graph = nx.read_edgelist('train_no_dup_no_header.csv', delimiter=',', nodetype=str)
    non_edges = nx.non_edges(graph)
    df = pd.DataFrame(list(non_edges))
    df['label'] = 0
    df = df.rename(columns={0: 'node1', 1: 'node2'})
    df2 = pd.read_csv('new_train_test.csv')
    df2['label'] = 1
    full_df = df2.append(df)
    print "test:\n\tnum of ones: {}\n\tnum of zeros:{}".format(df2.shape[0],df.shape[0])
    full_df.to_csv('train.csv', index=False)



def get_full_train_test(write_to_file=True):
    df = pd.read_csv('train_no_dup.csv')
    X_train, X_test = train_test_split(df, test_size=0.20)
    if write_to_file:
        X_train.to_csv('new_train_train.csv', index=False)
        X_train.to_csv('new_train_train_no_header.csv', index=False, header=False)
        X_test.to_csv('new_train_test.csv', index=False)
    else:
        return X_train, X_test


def check_for_hafifa():
    df1 = pd.read_csv('eval_test.csv')
    df2 = pd.read_csv('train.csv')
    comn_df = df1.append(df2)
    only_ones_df = comn_df[comn_df['label'] == 1]
    print only_ones_df.shape[0]
    comn_df[dup] = comn_df.drop_duplicates()
    print only_ones_df.shape[0]


def get_test_for_kaggle():
    graph = nx.read_edgelist('train_no_dup_no_header.csv', delimiter=',', nodetype=str)
    print "no dup: {}".format(graph.number_of_edges())
    # graph2 = nx.read_edgelist('train_no_header.csv', delimiter=',', nodetype=str)
    # print "with dup: {}".format(graph.number_of_edges())
    non_edges = nx.non_edges(graph)
    df = pd.DataFrame(list(non_edges))
    df = df.rename(columns={0: 'node1', 1: 'node2'})
    print "num of zeros:{}".format(df.shape[0])
    df.to_csv('./data_for_kaggle/kaggle_test.csv', index=False)


def remove_dup():
    graph = nx.read_edgelist('train_no_header.csv', delimiter=',', nodetype=str)
    edge_list = graph.edges()
    df = pd.DataFrame(list(edge_list))
    df = df.rename(columns={0: 'node1', 1: 'node2'})
    df.to_csv('train_no_dup.csv', index=False)
    df.to_csv('train_no_dup_no_header.csv', index=False, header=False)
    pass


def get_cross_val(k=10):
    train_path = "./data_for_testing/eval_test.csv"
    test_path = "./data_for_testing/eval_train.csv"

    experiments_dict = {}

    for i in range(k):
        print "making experiment {}".format(i)
        get_full_train_test()
        get_train()
        get_test()
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        experiments_dict[i] = (train,test)
    return experiments_dict

