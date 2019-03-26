from sklearn import svm
from sklearn import metrics


def run_svm(X_train, X_test, y_train, y_test):

    classifier = svm.SVC(C=1, gamma="scale", kernel="rbf", decision_function_shape="ovo")
    classifier.fit(X_train, y_train.values.ravel())

    y_predict = classifier.predict(X_test)
    print("SVM Accuracy: ", metrics.accuracy_score(y_test, y_predict))
