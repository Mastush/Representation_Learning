from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from lightning.classification import SDCAClassifier
import numpy as np

import utils
from model_wrapper import SVMWrapper


def get_classifier_type(classifier_type: str):
    options = {"sdca": SDCAClassifier, "sgd": SGDClassifier, "svc": SVC}
    if classifier_type.lower() in options:
        return options[classifier_type.lower()]
    raise TypeError("{} is not supported".format(classifier_type))


def get_linear_separator(x: np.ndarray, y: np.ndarray, return_classifier=True,
                         alpha: float = 0.0001, max_iter: int = 1000,
                         type_of_classifier: str = 'sdca', verbose: int = 1):
    if len(x.shape) > 2:
        x = utils.flatten_data(x)
    svm_classifier = get_classifier_type(type_of_classifier)
    if not hasattr(svm_classifier, "alpha"):
        svm_classifier = svm_classifier(C=alpha, max_iter=max_iter, verbose=verbose)
    else:
        svm_classifier = svm_classifier(alpha=alpha, max_iter=max_iter, verbose=verbose)
    svm_classifier.fit(x, y)
    if return_classifier:
        return SVMWrapper(svm_classifier)
    return svm_classifier.coef_

# TODO: implement partial training
