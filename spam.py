# coding: utf-8

# In[22]:

import numpy as np
import random
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

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

            X.append(item['CONTENT'])
            y.append(item['CLASS'])

    # Train/test szétválasztás
    split = 0.7
    data = np.asarray(X)
    perm = np.random.permutation(len(X))
    labels = np.asarray(y)
    perm2 = np.random.permutation(len(y))

    X_train = data[perm][0:int(len(data) * split)]
    X_test = data[perm][int(len(data) * split):]

    y_train = data[perm][0:int(len(data) * split)]
    y_test = data[perm][int(len(data) * split):]

    print('X Train set: ', np.shape(X_train))
    print('X Test set: ', np.shape(X_test))

    print('y Train set: ', np.shape(y_train))
    print('y Test set: ', np.shape(y_test))

    return X_test

def fit(X_train, y_train):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    text_clf.fit(X_train, y_train)

    return text_clf

# Buta osztályozó
def dumb_classify(data):
    threshold = 0.3
    if random.random() > threshold:
        return '1'
    else:
        return '0'


# Használd a 'train' adatokat az osztályozó módszer kidolgozására, a 'test' adatokat kiértékelésére!
# Lehetőleg használj gépi tanulást!
# Dokumentáld az érdekesnek tartott kísérleteket is!

# Példa kiértékelés 'recall' számításával.
# Kérdés: Milyen egyéb metrikát használnál kiértékelésre és miért?
def fit(test):

    sum_positive = 0
    found_positive = 0

    for datapoint in test:
        if datapoint[-1] == '1':
            sum_positive += 1
            if dumb_classify(datapoint) == '1':
                found_positive += 1

    print('Recall:', found_positive / sum_positive)

if __name__ == '__main__':
    fit(load_data())

