import pandas
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split


def load_data(data_file_name):
    data = pandas.read_csv(data_file_name, encoding = "ISO-8859-1")
    data = data.drop(labels=["filename"], axis=1)
    X = data.drop(labels=["label"], axis=1)
    Y = data.loc[:, data.columns.isin(['label'])]
    X = Normalizer(norm='l1').fit_transform(X)
    return X, Y


def get_test_train_set_split(X, Y, train_percent):
    return train_test_split(X, Y, test_size=1-train_percent, shuffle=True)


def get_test_train_set(file_name, train_percent):
    x, y = load_data(file_name)
    return get_test_train_set_split(x, y, train_percent)
