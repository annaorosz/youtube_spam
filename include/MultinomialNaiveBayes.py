# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


class MultinomialNaiveBayes(object):

    def __init__(self):
        pass

    # build a Mulinomial Naive Bayes model with the given X_train and y_train data
    def fit(self, X_train, y_train):
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
        text_clf.fit(X_train, y_train)

        joblib.dump(text_clf, 'MNB_model.h5')

