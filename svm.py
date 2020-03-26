from sklearn.linear_model import SGDClassifier

import utils


def get_linear_separator(x, y):
    if len(x.shape) > 2:
        x = utils.flatten_data(x)
    svm_classifier = SGDClassifier(fit_intercept=False)
    svm_classifier.fit(x, y)
    return svm_classifier.coef_
