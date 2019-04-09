from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def run_logit(X_train, X_test, y_train, y_test, is_multi_class, regularization=1, max_iter=20):
    classifier = LogisticRegression(solver='newton-cg', multi_class='ovr', C=regularization, max_iter=max_iter, n_jobs=-1)
    classifier.fit (X_train, y_train)
    test_out = classifier.predict(X_test)
    print (test_out)
    print (y_test)
    print("Logit (newton-cg) Accuracy: ", metrics.accuracy_score(y_test, test_out))

    classifier = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', multi_class='ovr', C=regularization, max_iter=max_iter)
    classifier.fit (X_train, y_train)
    test_out = classifier.predict(X_test)
    print(test_out)
    print (y_test)
    print("Logit (liblinear, l1 penalty) Accuracy: ", metrics.accuracy_score(y_test, test_out))

    classifier = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr', C=regularization, max_iter=max_iter)
    classifier.fit (X_train, y_train)
    test_out = classifier.predict(X_test)
    print(test_out)
    print (y_test)
    print("Logit (liblinear, l2 penalty) Accuracy: ", metrics.accuracy_score(y_test, test_out))

    '''
    classifier = LogisticRegression(solver='lbfgs', multi_class='auto', C=regularization, max_iter=max_iter, n_jobs=-1)
    classifier.fit (X_train, y_train.values.ravel())
    test_out = classifier.predict(X_test)
    print("Logit (lbfgs) Accuracy: ", metrics.accuracy_score(y_test, test_out))

    classifier = LogisticRegression(random_state=0, solver='sag', multi_class='auto', C=regularization, max_iter=max_iter, n_jobs=-1)
    classifier.fit (X_train, y_train.values.ravel())
    test_out = classifier.predict(X_test)
    print("Logit (sag) Accuracy: ", metrics.accuracy_score(y_test, test_out))

    classifier = LogisticRegression(random_state=0, penalty='l1', solver='saga', multi_class='auto', C=regularization, max_iter=max_iter, n_jobs=-1)
    classifier.fit (X_train, y_train.values.ravel())
    test_out = classifier.predict(X_test)
    print("Logit (saga, l1 penalty) Accuracy: ", metrics.accuracy_score(y_test, test_out))

    classifier = LogisticRegression(random_state=0, solver='saga', multi_class='auto', C=regularization, max_iter=max_iter, n_jobs=-1)
    classifier.fit (X_train, y_train.values.ravel())
    test_out = classifier.predict(X_test)
    print("Logit (saga, l2 penalty) Accuracy: ", metrics.accuracy_score(y_test, test_out))
    '''