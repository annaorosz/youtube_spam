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

def load_data():
    data = []
    X = []
    y = []

    # Betöltés
    with open('data.csv', 'r') as f:
        csvreader = csv.DictReader(f)
        for item in csvreader:
            # A 'data' tömb elemei: ['dátum string', 'szerző', 'komment', 'osztály cimke ('0': nem spam, '1': spam)']
            data.append([item['DATE'], item['AUTHOR'], item['CONTENT'], item['CLASS']])

            # a model epitesehez csak a kommentek szuksegesek
            X.append(item['CONTENT'])
            y.append(item['CLASS'])

    # Train/test szétválasztás
    split = 0.7
    data = np.asarray(X)
    labels = np.asarray(y)
    perm = np.random.permutation(len(X))

    X_train = data[perm][0:int(len(data) * split)]
    X_test = data[perm][int(len(data) * split):]

    y_train = labels[perm][0:int(len(labels) * split)]
    y_test = labels[perm][int(len(labels) * split):]

    print('X Train set: ', np.shape(X_train))
    print('X Test set: ', np.shape(X_test))

    print('y Train set: ', np.shape(y_train))
    print('y Test set: ', np.shape(y_test))

    return (X_train, y_train, X_test, y_test)

def fit_MNB(X_train, y_train):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    text_clf.fit(X_train, y_train)

    return text_clf

def fit_SGD(X_train, y_train):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,
                                               max_iter=5, tol=None))])
    text_clf.fit(X_train, y_train)

    return text_clf

def fit_SVM(X_train, y_train):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf', SVC(kernel=cosine_similarity, random_state=42))])
    text_clf.fit(X_train, y_train)

    return text_clf

def predict(clf, X_test, y_test):
    predicted = clf.predict(X_test)
    print (np.mean(predicted == y_test))
    print(metrics.classification_report(y_test, predicted))


def gs_SGD(X_train, y_train, X_test, y_test):
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],  'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3, 1e-4),
                  'clf__loss': ('log', 'perceptron'),  'clf__max_iter': (3, 4, 5), 'clf__shuffle': (True, False),}
    gs_clf = GridSearchCV(fit_SGD(X_train, y_train), parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(X_train, y_train)
    predicted = gs_clf.predict(X_test)

    print (np.mean(predicted == y_test))
    print(metrics.classification_report(y_test, predicted))
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def gs_SVM(X_train, y_train, X_test, y_test):
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],  'tfidf__use_idf': (True, False),
                  'clf__kernel': ('rbf', cosine_similarity, 'linear', 'poly'),}
    gs_clf = GridSearchCV(fit_SVM(X_train, y_train), parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(X_train, y_train)
    predicted = gs_clf.predict(X_test)

    print (np.mean(predicted == y_test))
    print(metrics.classification_report(y_test, predicted))
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))



# Használd a 'train' adatokat az osztályozó módszer kidolgozására, a 'test' adatokat kiértékelésére!
# Lehetőleg használj gépi tanulást!
# Dokumentáld az érdekesnek tartott kísérleteket is!

# Példa kiértékelés 'recall' számításával.
# Kérdés: Milyen egyéb metrikát használnál kiértékelésre és miért?

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()

    #fit with a Multinomial Naive Bayes model
    clf = fit_MNB(X_train, y_train)
    predict(clf, X_test, y_test)

    #fit with an SVM gradient descent classifier
    clf = fit_SGD(X_train, y_train)
    predict(clf, X_test, y_test)

    #fit with an SVM cosine similarity classifier
    clf = fit_SVM(X_train, y_train)
    predict(clf, X_test, y_test)

    #grid search for sgd
    gs_SGD(X_train, y_train, X_test, y_test)

    # grid search for cos
    gs_SVM(X_train, y_train, X_test, y_test)