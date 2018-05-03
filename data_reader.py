import random

import networkx as nx

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
    return graph
