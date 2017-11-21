# coding: utf-8

# Author: Anna Orosz
# Nov 2017
#
# main.py script that calls each of the following classes/methods:
# call DataLoader to get training and testing data
# call MultinomialNaiveBayes to build an MNB model and save it
# call SupportVectorMachine to build and SVM model and save it
# once models are built and saved:
# call the Classifier method to classify a given test set
# and analyze results
#
# optional: call the grid-search method to find best possible
# parameters for the MNB and SVM models in their respective classes
#
# outputs: the recall and f1 scores for the MNB and SVM models


import sys

from include.DataLoader import *
from include.MultinomialNaiveBayes import *
from include.SupportVectorMachine import *
from include.Classifier import *
from sklearn.externals import joblib


if __name__ == '__main__':

    # change this depending on the location of file when executed
    sys.path.append("/home/logmein/hw")

    # initialize a DataLoader object to get the training and testing data
    data_loader = DataLoader()
    X_train, y_train, X_test, y_test = data_loader.load('data/data.csv')

    # classifier object that takes a model and makes a prediction
    clf = Classifier()

    #build and predict with the MNB model
    print "Training and Testing the Multinomial Naive Bayes model..."
    MultinomialNaiveBayes().fit(X_train, y_train)
    clf.predict(joblib.load('models/MNB_model.h5'), X_test, y_test)

    # build and predict with the SVM model
    print "Training and Testing the Support Vector Machine model..."
    SupportVectorMachine().fit(X_train, y_train)
    clf.predict(joblib.load('models/SVM_model.h5'), X_test, y_test)


    # gridsearching for the best parameters for MNB and SVM models
    # not necessary for final solution
    # commented out because execution is expensive

    MultinomialNaiveBayes().gs(X_train, y_train, X_test, y_test)
    SupportVectorMachine().gs(X_train, y_train, X_test, y_test)
