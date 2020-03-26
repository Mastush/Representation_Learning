from sklearn.linear_model import SGDClassifier


def get_linear_separator(x, y):
    svm_classifier = SGDClassifier(fit_intercept=False)
    svm_classifier.fit(x, y)
    return svm_classifier.coef_
