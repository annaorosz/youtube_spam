# coding: utf-8

# Author: Anna Orosz
# Nov 2017
#
# Classifier class with the predict() function
# predict() takes in a model which is being loaded in main.py
#   classifies each instance in the test data set
#   compares expected and predicted results
#   uses recall and f1 scores to analyze the difference between these


import numpy as np
from sklearn import metrics
from sklearn import preprocessing


class Classifier(object):

    def __init__(self):
        pass


    #predict with the given model (MNB, SGD or SVM) for the given X_test data and compare results to expected y_test
    #returns the accuracy for the given model by calculating the difference between the predicted and expected labels
    def predict(self, clf, X_test, y_test):

        # binarize the labels so that each is 0 or 1
        lb = preprocessing.LabelBinarizer()
        y_test = np.array([number[0] for number in lb.fit_transform(y_test)])

        # predict with the testing data
        predicted = clf.predict(X_test)

        # Példa kiértékelés 'recall' számításával.
        # the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.
        # the ability of the classifier to find all the positive samples.
        print ("recall score = %f" %(metrics.recall_score(y_test, predicted)))

        # f1 is hasznos a pelda kiertekelesere
        # masik neven 'harmonic mean', mert egyfajta atlag a recall es a precision kozott
        # sokszor hasznaljak NLP teruleten
        # a weighted average of the precision and recall: f1 = 2 * (precision * recall) / (precision + recall)
        print ("f1 score = %f" % (metrics.f1_score(y_test, predicted)))
