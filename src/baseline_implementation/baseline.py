from src.baseline_implementation import svm_implementaion
from src.baseline_implementation import logit_implementation


def run_all_baselines(X_train, X_test, y_train, y_test, is_multi_class):
    logit_implementation.run_logit(X_train, X_test, y_train, y_test, is_multi_class)
    svm_implementaion.run_svm(X_train, X_test, y_train, y_test, 3, 73, 10)
