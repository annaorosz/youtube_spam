# coding: utf-8

import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV


#load the data from data.csv
def load_data():
    '''
    X = array of the "content" feature in the data
    y = label for each comment, 0 or 1
    :return: tuple of data and label for both the training and testing
    '''

    X = []
    y = []

    # Betöltés
    with open('data.csv', 'r') as f:
        csvreader = csv.DictReader(f)
        for item in csvreader:
            # a model epitesehez csak a kommentek szuksegesek
            X.append(item['CONTENT'])
            y.append(item['CLASS'])

    # Train/test szétválasztás
    # 'train' adatokat az osztályozó módszer kidolgozására, a 'test' adatokat kiértékelésére
    split = 0.7
    data = np.asarray(X)
    labels = np.asarray(y)
    perm = np.random.permutation(len(X))

    X_train = data[perm][0:int(len(data) * split)]
    X_test = data[perm][int(len(data) * split):]

    y_train = labels[perm][0:int(len(labels) * split)]
    y_test = labels[perm][int(len(labels) * split):]

    return (X_train, y_train, X_test, y_test)


# build a Mulinomial Naive Bayes model with the given X_train and y_train data
def fit_MNB(X_train, y_train):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    text_clf.fit(X_train, y_train)

    return text_clf


# build a Stochastic Gradient Descent model with the given X_train and y_train data
def fit_SGD(X_train, y_train):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(max_iter=5, tol=None))])
    text_clf.fit(X_train, y_train)

    return text_clf


# build an SVM model with a cosine similarity kernel with the given X_train and y_train data
def fit_SVM(X_train, y_train):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf', SVC(kernel=cosine_similarity))])
    text_clf.fit(X_train, y_train)

    return text_clf


# build the best model pre-computed using grid search
def fit_best_model(X_train, y_train):
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))), ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf', SVC(kernel=cosine_similarity, random_state=42))])
    text_clf.fit(X_train, y_train)

    return text_clf


#predict with the given model (MNB, SGD or SVM) for the given X_test data and compare results to expected y_test
#returns the accuracy for the given model by calculating the difference between the predicted and expected labels
def predict(clf, X_test, y_test):
    predicted = clf.predict(X_test)

    #print(metrics.classification_report(y_test, predicted))

    # Példa kiértékelés 'recall' számításával.
    return np.mean(predicted == y_test)


# tune the parameters for the SGD model using Grid Search and print the metrics for the best such model
# not executing in the final version because the computation is expensive / takes a long time
# csak a kiserlethez szukseges, nem a vegso verziohoz
def gs_SGD(X_train, y_train, X_test, y_test):
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3, 1e-4), 'clf__loss': ('log', 'perceptron'),  'clf__max_iter': (3, 4, 5),
                  'clf__shuffle': (True, False), 'clf__random_state': (42, None),}

    gs_clf = GridSearchCV(fit_SGD(X_train, y_train), parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(X_train, y_train)
    predicted = gs_clf.predict(X_test)

    print(metrics.classification_report(y_test, predicted))
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    # Példa kiértékelés 'recall' számításával.
    print np.mean(predicted == y_test)


# tune the parameters for the SVM model using Grid Search and print the metrics for the best such model
# not executing in the final version because the computation is expensive / takes a long time
# csak a kiserlethez szukseges, nem a vegso verziohoz
def gs_SVM(X_train, y_train, X_test, y_test):
    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],  'tfidf__use_idf': (True, False),
                  'clf__kernel': ('rbf', cosine_similarity, 'linear', 'poly'), 'clf__random_state': (42, None),}

    gs_clf = GridSearchCV(fit_SVM(X_train, y_train), parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(X_train, y_train)
    predicted = gs_clf.predict(X_test)

    print(metrics.classification_report(y_test, predicted))
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    # Példa kiértékelés 'recall' számításával.
    print np.mean(predicted == y_test)


# Kérdés: Milyen egyéb metrikát használnál kiértékelésre és miért?

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()

    #fit with a Multinomial Naive Bayes model
    clf = fit_MNB(X_train, y_train)
    print "Multinomial Naive Bayes model accuracy:"
    print predict(clf, X_test, y_test)
    print "\n"

    #fit with an SVM gradient descent classifier
    clf = fit_SGD(X_train, y_train)
    print "Stochastic Gradient Descent model accuracy:"
    print predict(clf, X_test, y_test)
    print "\n"

    #fit with an SVM cosine similarity classifier
    clf = fit_SVM(X_train, y_train)
    print "Support Vector Machine model accuracy:"
    print predict(clf, X_test, y_test)
    print "\n"

    #fit with the best model found by grid search
    clf = fit_best_model(X_train, y_train)
    print "Best model accuracy:"
    print predict(clf, X_test, y_test)
    print "\n"

    #grid search for SGD
    #gs_SGD(X_train, y_train, X_test, y_test)

    # grid search for SVM
    gs_SVM(X_train, y_train, X_test, y_test)
