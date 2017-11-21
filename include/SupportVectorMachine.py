# coding: utf-8

# Author: Anna Orosz
# Nov 2017
#
# MultinomialNaiveBayes class with two relevant methods
# gs() computes the best parameters to build the most accurate method
#   prints the metrics for said best method
#   prints the best values for each given parameter
# fit() builds the best model with the values that were calculated in
# gs() prior saves this model as MNB_model.h5


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

        # binarize the labels so that each is 0 or 1
        lb = preprocessing.LabelBinarizer()
        y_train = np.array([number[0] for number in lb.fit_transform(y_train)])

        # set the parameters to precomputed values
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))), ('tfidf', TfidfTransformer(use_idf=True)),
                             ('clf', SVC(kernel=cosine_similarity, random_state=42))])
        text_clf.fit(X_train, y_train)

        # save model as a .h5 file
        joblib.dump(text_clf, 'models/SVM_model.h5')


    # tune the parameters for the SVM model using Grid Search and print the metrics for the best such model
    # not executing in the final version because the computation is expensive / takes a long time
    # csak a kiserlethez szukseges, nem a vegso verziohoz
    def gs(self, X_train, y_train, X_test, y_test):

        # different parameters to try in the Grid Search
        parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)], 'tfidf__use_idf': (True, False),
                      'clf__kernel': ('rbf', cosine_similarity, 'linear', 'poly'), 'clf__random_state': (42, None), }

        self.fit(X_train, y_train)
        gs_clf = GridSearchCV(joblib.load('models/SVM_model.h5'), parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(X_train, y_train)
        predicted = gs_clf.predict(X_test)

        # print best parameters and the metrics for the best model
        print(metrics.classification_report(y_test, predicted))
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
