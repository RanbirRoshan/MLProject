import pandas
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import numpy


def just_load_data (data_file_name):
    data = pandas.read_csv(data_file_name, encoding = "ISO-8859-1")
    return shuffle(data)


def load_raw_data(data_file_name):
    data = just_load_data(data_file_name)
    data = data.drop(labels=["filename"], axis=1)
    X = data.drop(labels=["label"], axis=1)
    Y = data.loc[:, data.columns.isin(['label'])]
    return X, Y


def load_data(data_file_name, apply_feature_selection=False):
    data = pandas.read_csv(data_file_name, encoding = "ISO-8859-1")
    data = data.drop(labels=["filename"], axis=1)
    data = shuffle(data)
    X, Y = load_raw_data(data_file_name)
    X = data.drop(labels=["label","LSTMFileLoc"], axis=1).as_matrix()
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)
    X = Normalizer(norm='l1').fit_transform(X)
    X = get_feature_importance_val(X, Y, apply_feature_selection)
    is_multi_class = False
    if data.label.unique().shape[0]>2:
        is_multi_class = True
    return X, Y, is_multi_class


def get_test_train_set_split(X, Y, train_percent):
    return train_test_split(X, Y, test_size=1-train_percent, shuffle=True)


def get_test_train_set(file_name, train_percent, apply_feature_selection=False):
    x, y, is_multi_class = load_data(file_name, apply_feature_selection)
    X_train, X_test, y_train, y_test = get_test_train_set_split(x, y, train_percent)
    return X_train, X_test, y_train, y_test, is_multi_class


def get_feature_importance_val(X, Y, apply_feature_selection):
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, Y)
    print("Feature Importance Values:")
    print(clf.feature_importances_)
    if not apply_feature_selection:
        return X
    print("Original Dimension: ", X.shape)
    X = SelectFromModel(clf, prefit=True).transform(X)
    print("Feature reduced Dimension: ", X.shape)
    return X


def load_LSTM_file(file_name):
    data = just_load_data(file_name)
    data = data[:3421]
    data = Normalizer(norm='l1').fit_transform(data)
    return data.T


def load_LSTM_data(file_name, train_percent, folder_name, apply_feature_selection=False):
    X, Y = load_raw_data(file_name)
    X = X.loc[:, X.columns.isin(['LSTMFileLoc'])]
    d = load_LSTM_file(folder_name + "/" + X.LSTMFileLoc[0])
    z = numpy.zeros((X.shape[0],d.shape[0],d.shape[1]))
    for i in range (0, X.shape[0]):
        file = folder_name + "/" + X.LSTMFileLoc[i]
        d = load_LSTM_file(file)
        z[i]=d
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)
    #X_train, X_test, y_train, y_test = get_test_train_set_split(z, Y, train_percent)
    is_multi_class = False
    #if Y.unique().shape[0]>2:
    #    is_multi_class = True
    return z, None, Y, None, is_multi_class
