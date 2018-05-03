import random
import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.model_selection import train_test_split

DATA_SIZE = 1434
path = "train.csv"


def split_to_train_and_test(full_graph, num_of_edge_to_remove):
    graph = full_graph.copy()
    test_graph = nx.Graph()
    test_graph.add_nodes_from(range(1434))
    edges = list(graph.edges())
    to_remove = random.sample(edges, num_of_edge_to_remove)

    for chosen_edge in to_remove:
        try:
            chosen_edge = random.choice(edges)
            graph.remove_edge(chosen_edge[0], chosen_edge[1])
            test_graph.add_edge(chosen_edge[0], chosen_edge[1])
        except nx.exception.NetworkXError:
            continue
    return graph, test_graph

def get_train_test():
    kfold_num = 10
    graph = nx.read_edgelist(path, delimiter=',', nodetype=str)
    print "number of nodes: {}".format(graph.number_of_nodes())
    print "number of ednges: {}".format(graph.number_of_edges())
    num_of_edges = graph.number_of_edges()
    dict_of_train_test = {}
    edge_to_remove = num_of_edges / kfold_num
    train, test = split_to_train_and_test(graph, edge_to_remove)
    return train, test


def get_cross_val_train_test(kfold_num):
    graph = nx.read_edgelist(path, delimiter=',', nodetype=str)
    print "number of nodes: {}".format(graph.number_of_nodes())
    print "number of ednges: {}".format(graph.number_of_edges())
    num_of_edges = graph.number_of_edges()
    dict_of_train_test = {}
    for k in range(kfold_num):
        edge_to_remove = num_of_edges / kfold_num
        train, test = split_to_train_and_test(graph, edge_to_remove)
        dict_of_train_test[k] = (train, test)


def get_nx_graph():
    kfold_num = 10
    graph = nx.read_edgelist(path, delimiter=',', nodetype=str)
    print "number of nodes: {}".format(graph.number_of_nodes())
    print "number of ednges: {}".format(graph.number_of_edges())
    non_edges = nx.non_edges(graph)
    df = pd.DataFrame(list(non_edges))
    df['label'] = 0
    df = df.rename(columns={0: 'node1', 1: 'node2'})
    df2 = pd.read_csv(path)
    df2['label'] = 1
    full_df = df2.append(df)
    full_df.to_csv('full_train.csv', index=False)

    return graph

def get_non_edges():
    pass

def get_train_test_as_edge_list():
    train, test = get_train_test()
    nx.write_edgelist(train, 'train_train.csv')
    nx.write_edgelist(test, 'train_test.csv')

def read_sparse_matrix(train_data):
    print 'creating matrix'
    x = train_data['node1']
    y = train_data['node2']
    assert len(x) == len(y)
    data = [1] * len(x)
    m = csr_matrix((data,(x,y)), shape=(DATA_SIZE,DATA_SIZE),dtype='f')
    print 'm shape:', m.shape, 'm non zeros:', m.nnz
    assert np.allclose(m.todense(), m.T.todense(), atol=1e-8) #matrix is symmetric
    return m.todense()#the matrix is small, sparse matrix is not necessary.

def create_holdout_set(train_path,train_percent = 0.9):
    df = pd.read_csv(train_path)
    m_train = read_sparse_matrix(df)
    #create holdout set. the set will contains both existing and non-existing edges.
    m_train_holdout = np.matrix(m_train)
    validation_set = set()
    for i in range(DATA_SIZE):
        for j in range(i+1, DATA_SIZE):
            if random.random() > train_percent:
                validation_set.add((i, j))
                m_train_holdout[i, j] = 0
                m_train_holdout[j, i] = 0
    return m_train_holdout, validation_set


def get_full_train_test():
    df = pd.read_csv('full_train.csv')
    X_train, X_test = train_test_split(df, test_size=0.33, random_state=42)
    X_train.to_csv('train_train.csv', index=False)
    X_test.to_csv('train_test.csv', index=False)
# create_holdout_set(path)
get_full_train_test()
