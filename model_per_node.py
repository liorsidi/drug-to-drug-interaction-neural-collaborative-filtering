import pandas as pd


def feature_dict_to_df(med_feature_dict):
    """
    feature_dict (dict) - { 'm0' : { 'f1' : 1, 'f2' : 2 } }
    :return: pandas dataframe: node_name, f1, f2, f3, ..., label
    """
    node_names = med_feature_dict.keys()
    feature_names = med_feature_dict[node_names[0]].keys()
    data_mock = {
        'node_name': node_names
    }
    for f_idx, f_name in enumerate(feature_names):
        f_column = [med_feature_dict[med].values()[f_idx] for med in med_feature_dict]
        # d = { 'node_name': ['m1', 'm2'], 'f1': [3, 4], 'f2': [4, 5] }
        data_mock[f_name] = f_column
    df = pd.DataFrame(data=data_mock)

    return df


def extract_med_features(graph):
    """

    :return:
    """
    med_feature_dict_mock = {
        'm0': {
            'f1': 1,
            'f2': 2,
            'f3': 3
        },
        'm1': {
            'f1': 4,
            'f2': 5,
            'f3': 6
        },
        'm2': {
            'f1': 7,
            'f2': 8,
            'f3': 9
        }
    }
    return med_feature_dict_mock


def get_graph():
    # TODO
    return 'placeholder'


def go():
    graph = get_graph()
    med_features = extract_med_features(graph)
    df = feature_dict_to_df(med_features)
    # remove 'm1'
    x = 1

go()