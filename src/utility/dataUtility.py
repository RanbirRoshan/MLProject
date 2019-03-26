import pandas
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def load_data(data_file_name, apply_feature_selection=False):
    data = pandas.read_csv(data_file_name, encoding = "ISO-8859-1")
    data = data.drop(labels=["filename"], axis=1)
    X = data.drop(labels=["label"], axis=1)
    Y = data.loc[:, data.columns.isin(['label'])]
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
    clf = clf.fit(X, Y.values.ravel())
    print("Feature Importance Values:")
    print(clf.feature_importances_)
    if not apply_feature_selection:
        return X
    print("Original Dimension: ", X.shape)
    X = SelectFromModel(clf, prefit=True).transform(X)
    print("Feature reduced Dimension: ", X.shape)
    return X
