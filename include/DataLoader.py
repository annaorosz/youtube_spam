# coding: utf-8

import numpy as np
import csv

class DataLoader(object):

    def __init__(self):
        pass

    #load the data from data.csv
    def load(self):
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

