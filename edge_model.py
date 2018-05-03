import networkx as nx
import numpy as np
import pandas as pd
import os
import random
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
random.seed(4)


class DMBI_hackathon_ddi_utils():
    NODE_1 = 'node1'
    NODE_2 = 'node2'

    def __init__(self, number_of_drugs=1434):
        self.number_of_drugs = number_of_drugs

    def write_list_to_file(self, list, path):
        thefile = open(path, 'w')
        for item in list:
            thefile.write("%s\n" % item)
        thefile.close()

    def read_sparse_matrix(self, train_data):
        print 'creating matrix'
        x = train_data[self.NODE_1]
        y = train_data[self.NODE_2]
        assert len(x) == len(y)
        data = [1] * len(x)
        m = csr_matrix((data, (x, y)), shape=(self.number_of_drugs, self.number_of_drugs), dtype='f')
        print 'm shape: {0} m non zeros: {1}'.format(m.shape, m.nnz)
        assert np.allclose(m.todense(), m.T.todense(), atol=1e-8)  # matrix is symmetric
        return m.todense()  # the matrix is small, sparse matrix is not necessary.

    def write_solution_to_file(self, preds, file_path, num_interactions_train):
        # preds is assumed to be ordered by confidence level
        # adds the header to the solution, combines the node IDs and writes the solution to file
        # asserts are important. Note them.

        print 'writing predictions to file: {0}'.format(file_path)
        for u, v in preds:
            assert u < v, 'graph is undirected, predict edges where the first node id is smaller than the second only'
        assert len(preds) == (
                    self.number_of_drugs * self.number_of_drugs - self.number_of_drugs - num_interactions_train) / 2, "number of predictions is equal to number of non existing edges"
        output = [','.join([self.NODE_1 + '_' + self.NODE_2])] + [','.join([str(p[0]) + '_' + str(p[1])]) for p in
                                                                  preds]
        self.write_list_to_file(output, file_path)

    def create_holdout_set(self, m_train, train_percent=0.9):
        # create holdout set. the set will contains both existing and non-existing edges.
        m_train_holdout = np.matrix(m_train)
        validation_set = set()
        for i in range(self.number_of_drugs):
            for j in range(i + 1, self.number_of_drugs):
                if random.random() > train_percent:
                    validation_set.add((i, j))
                    m_train_holdout[i, j] = 0
                    m_train_holdout[j, i] = 0
        return m_train_holdout, validation_set

    def average_precision_at_k(self, k, class_correct):
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

# simple prediction class
class link_prediction_predictor:
    def __init__(self, number_of_drugs):
        self.G = nx.Graph()
        self.G.add_nodes_from(range(number_of_drugs))

    def fit(self, edge_list):
        self.G.add_edges_from(edge_list)
        edge_features = self.extract_features(edge_list)


    def predict(self, prediction_set=None):
        self.extract_features(prediction_set)
        # if prediction_set is None then all non-existent edges in the graph will be used.



        # Communities
        # https://networkx.github.io/documentation/stable/reference/algorithms/community.html

        # https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.link_prediction.cn_soundarajan_hopcroft.html#networkx.algorithms.link_prediction.cn_soundarajan_hopcroft
        # common_neighbords = nx.cn_soundarajan_hopcroft(self.G, ebunch=prediction_set)

        # https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.link_prediction.within_inter_cluster.html#networkx.algorithms.link_prediction.within_inter_cluster
        # nx.ra_index_soundarajan_hopcroft(self.G, )

        # https: // networkx.github.io / documentation / networkx - 1.10 / reference / generated / networkx.algorithms.link_prediction.within_inter_cluster.html  # networkx.algorithms.link_prediction.within_inter_cluster
        # nx.within_inter_cluster(...  )

        # predictions are expected as described in write_solution_to_file. The values are suppose to be ordered by confidence.
        return preds

    def extract_features(self, prediction_set):
        # res_alloc_vals = [(u1, v1) for (p, u1, v1) in
        #          sorted([(prediction, u, v) for (u, v, prediction) in preds], reverse=True)]
        # if ebunch is None then all non-existent edges in the graph will be used.
        edge_features = defaultdict(dict)
        res_alloc = nx.resource_allocation_index(self.G, ebunch=prediction_set)
        self.append_features(edge_features, feature_name='res_alloc', tufeature_list=res_alloc)

        jaccard_coef = nx.jaccard_coefficient(self.G, ebunch=prediction_set)
        self.append_features(edge_features, feature_name='jaccard_coef', feature_list=jaccard_coef)

        adamic_adar = nx.adamic_adar_index(self.G, ebunch=prediction_set)
        self.append_features(edge_features, feature_name='adamic_adar', feature_list=adamic_adar)

        pref_attachment = nx.preferential_attachment(self.G, ebunch=prediction_set)
        self.append_features(edge_features, feature_name='pref_attachment', feature_list=pref_attachment)

        # to node list and feature lists
        edge_names = edge_features.keys()
        feature_names = edge_features[edge_names[0]].keys()
        data_dict = {
            'node_name': edge_names
        }
        for f_idx, f_name in enumerate(feature_names):
            f_column = [edge_features[edge].values()[f_idx] for edge in edge_features]
            # d = { 'node_name': ['m1', 'm2'], 'f1': [3, 4], 'f2': [4, 5] }
            data_mock[f_name] = f_column
        df = pd.DataFrame(data=data_mock)

        return edge_features

    def append_features(self, edge_features, feature_name, feature_list):
        for u, v, score in feature_list:
            edge_features[(u, v)][feature_name] = score


DMBI_hackathon_ddi = DMBI_hackathon_ddi_utils()
train_matrix = DMBI_hackathon_ddi.read_sparse_matrix(pd.read_csv('train.csv'))

# Evaluate model. #Evalua
# Note that holdout is based on random decision.
# Test set contains new interactions that random selection does not emulate.

m_train_holdout, validation_set = DMBI_hackathon_ddi.create_holdout_set(train_matrix)
x, y = m_train_holdout.nonzero()  # x and y indices of nonzero cells (existing edges)
edge_list = list(zip(x, y))
link_prediction = link_prediction_predictor(DMBI_hackathon_ddi.number_of_drugs)
link_prediction.fit(edge_list)
preds = link_prediction.predict(validation_set)
class_correct = [train_matrix[x[0], x[1]] for x in preds]
average_precision = DMBI_hackathon_ddi.average_precision_at_k(k=100, class_correct=class_correct)
print 'average precision @ 100: {0}'.format(average_precision)


#Create final submission file#Create
x,y = train_matrix.nonzero()
num_interactions_train = len(x);assert len(x)==len(y)
edge_list = list(zip(x,y))
link_prediction = link_prediction_predictor(DMBI_hackathon_ddi.number_of_drugs)
link_prediction.fit(edge_list)
preds = link_prediction.predict()
DMBI_hackathon_ddi.write_solution_to_file(preds,'sample_predictions.csv',num_interactions_train=num_interactions_train)