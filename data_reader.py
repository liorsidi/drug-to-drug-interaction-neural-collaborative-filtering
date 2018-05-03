import networkx as nx

path = "train.csv"



def read_as_network_x():
    graph = nx.read_edgelist(path, delimiter=',', nodetype=str)
    print "number of nodes: {}".format(graph.number_of_nodes())
    return graph

