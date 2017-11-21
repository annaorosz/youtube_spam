# coding: utf-8

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.externals import joblib
from sklearn import preprocessing


class SupportVectorMachine(object):

    def __init__(self):
        pass


    # build an SVM model with a cosine similarity kernel with the given X_train and y_train data
    def fit(self, X_train, y_train):
        lb = preprocessing.LabelBinarizer()
        y_train = np.array([number[0] for number in lb.fit_transform(y_train)])

        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))), ('tfidf', TfidfTransformer(use_idf=True)),
                             ('clf', SVC(kernel=cosine_similarity, random_state=42))])
        text_clf.fit(X_train, y_train)

        joblib.dump(text_clf, 'SVM_model.h5')


    # tune the parameters for the SVM model using Grid Search and print the metrics for the best such model
    # not executing in the final version because the computation is expensive / takes a long time
    # csak a kiserlethez szukseges, nem a vegso verziohoz
    def gs_SVM(self, X_train, y_train, X_test, y_test):
        parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)], 'tfidf__use_idf': (True, False),
                      'clf__kernel': ('rbf', cosine_similarity, 'linear', 'poly'), 'clf__random_state': (42, None), }

        self.fit(X_train, y_train)
        gs_clf = GridSearchCV(joblib.load('SVM_model.h5'), parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(X_train, y_train)
        predicted = gs_clf.predict(X_test)

        print(metrics.classification_report(y_test, predicted))
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
