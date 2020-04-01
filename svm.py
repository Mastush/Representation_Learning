from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
import numpy as np

import utils
from model_wrapper import SVMWrapper


def get_linear_separator(x: np.ndarray, y: np.ndarray, return_classifier=True):
    if len(x.shape) > 2:
        x = utils.flatten_data(x)
    svm_classifier = SGDClassifier(loss='squared_hinge')
    svm_classifier.fit(x, y)
    if return_classifier:
        return SVMWrapper(svm_classifier)
    return svm_classifier.coef_
