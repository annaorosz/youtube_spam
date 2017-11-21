import sys

from include.DataLoader import *
from include.MultinomialNaiveBayes import *
from include.SupportVectorMachine import *
from include.Classifier import *
from sklearn.externals import joblib


if __name__ == '__main__':

    # change this depending on the location of file when executed
    sys.path.append("/home/logmein/hw")

    data_loader = DataLoader()
    X_train, y_train, X_test, y_test = data_loader.load()

    clf = Classifier()

    MultinomialNaiveBayes().fit(X_train, y_train)
    print clf.predict(joblib.load('MNB_model.h5'), X_test, y_test)

    SVM = SupportVectorMachine().fit(X_train, y_train)
    print clf.predict(joblib.load('SVM_model.h5'), X_test, y_test)
