from sklearn import svm
from sklearn import metrics


def run_svm(X_train, X_test, y_train, y_test, poly_degree_start=3, poly_degree_end=3, step=1):

    classifier = svm.SVC(C=1, gamma="scale", kernel="rbf", decision_function_shape="ovo")
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = rbf) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    classifier = svm.SVC(C=1, gamma="scale", kernel="poly", decision_function_shape="ovo")
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = poly) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    classifier = svm.SVC(C=1, gamma="scale", kernel="linear", decision_function_shape="ovo")
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = linear) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    classifier = svm.SVC(C=1, gamma="scale", kernel="sigmoid", decision_function_shape="ovo")
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = sigmoid) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    classifier = svm.SVC(C=1, gamma="scale", kernel="rbf", decision_function_shape="ovo", probability=True)
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = rbf, prior probability) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    for i in range(poly_degree_start, poly_degree_end+1, step):
        classifier = svm.SVC(C=1, gamma="scale", kernel="poly", decision_function_shape="ovo", probability=True, degree=i)
        classifier.fit(X_train, y_train.values.ravel())
        y_predict = classifier.predict(X_test)
        print("SVM (kernel = poly, prior probability, degree "+str(i)+") Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    classifier = svm.SVC(C=1, gamma="scale", kernel="linear", decision_function_shape="ovo", probability=True)
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = linear, prior probability) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    classifier = svm.SVC(C=1, gamma="scale", kernel="sigmoid", decision_function_shape="ovo", probability=True)
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = sigmoid, prior probability) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    classifier = svm.SVC(C=1, gamma="scale", kernel="rbf", decision_function_shape="ovr")
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = rbf, ovr) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    classifier = svm.SVC(C=1, gamma="scale", kernel="poly", decision_function_shape="ovr")
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = poly, ovr) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    classifier = svm.SVC(C=1, gamma="scale", kernel="linear", decision_function_shape="ovr")
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = linear, ovr) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    classifier = svm.SVC(C=1, gamma="scale", kernel="sigmoid", decision_function_shape="ovr")
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = sigmoid, ovr) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    classifier = svm.SVC(C=1, gamma="scale", kernel="rbf", decision_function_shape="ovr", probability=True)
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = rbf, prior probability, ovr) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    for i in range(poly_degree_start, poly_degree_end+1, step):
        classifier = svm.SVC(C=1, gamma="scale", kernel="poly", decision_function_shape="ovr", probability=True, degree=i)
        classifier.fit(X_train, y_train.values.ravel())
        y_predict = classifier.predict(X_test)
        print("SVM (kernel = poly, prior probability, degree "+str(i)+", ovr) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    classifier = svm.SVC(C=1, gamma="scale", kernel="linear", decision_function_shape="ovr", probability=True)
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = linear, prior probability, ovr) Accuracy: ", metrics.accuracy_score(y_test, y_predict))

    classifier = svm.SVC(C=1, gamma="scale", kernel="sigmoid", decision_function_shape="ovr", probability=True)
    classifier.fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)
    print("SVM (kernel = sigmoid, prior probability, ovr) Accuracy: ", metrics.accuracy_score(y_test, y_predict))
