# coding: utf-8

import numpy as np
from sklearn import metrics
from sklearn import preprocessing


class Classifier(object):

    def __init__(self):
        pass


    #predict with the given model (MNB, SGD or SVM) for the given X_test data and compare results to expected y_test
    #returns the accuracy for the given model by calculating the difference between the predicted and expected labels
    def predict(self, clf, X_test, y_test):

        lb = preprocessing.LabelBinarizer()
        y_test = np.array([number[0] for number in lb.fit_transform(y_test)])

        predicted = clf.predict(X_test)

        # Példa kiértékelés 'recall' számításával.
        print ("recall score = %f" %(metrics.recall_score(y_test, predicted)))
        print ("f1 score = %f" % (metrics.f1_score(y_test, predicted)))
