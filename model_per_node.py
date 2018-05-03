import pandas as pd


def feature_dict_to_df():
    """
    feature_dict (dict) - { 'm0' : { 'f1' : 1, 'f2' : 2 } }
    :return: pandas dataframe: node_name, f1, f2, f3, ..., label
    """
    feature_dict_mock = {
        'm0': {
            'f1': 1,
            'f2': 2,
            'f3': 3
        },
        'm1': {
            'f1': 1,
            'f2': 2,
            'f3': 3
        },
        'm2': {
            'f1': 1,
            'f2': 2,
            'f3': 3
        }
    }
    node_names = feature_dict_mock.keys()
    feature_names = feature_dict_mock[node_names[0]].keys()
    feature_values = feature_dict_mock[node_names[0]].values()
    data_mock = {'node_name': node_names }
    for i in range(len(feature_names)):
        f_name, f_val = feature_names[i], feature_values[i]


    df = pd.DataFrame(data=data_mock)
    return df


def extract_features():
    """

    :return:
    """
