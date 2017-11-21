# coding: utf-8

import numpy as np

class Classifier(object):

    def __init__(self):
        pass

    #predict with the given model (MNB, SGD or SVM) for the given X_test data and compare results to expected y_test
    #returns the accuracy for the given model by calculating the difference between the predicted and expected labels
    def predict(self, clf, X_test, y_test):
        predicted = clf.predict(X_test)

        #print(metrics.classification_report(y_test, predicted))

        # Példa kiértékelés 'recall' számításával.
        return np.mean(predicted == y_test)
