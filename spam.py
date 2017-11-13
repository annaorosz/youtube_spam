# coding: utf-8

# In[22]:

import numpy as np
import random
import csv

def load_data():

    data = []

    # Betöltés
    with open('data.csv', 'r') as f:
        csvreader = csv.DictReader(f)
        for item in csvreader:
            data.append([item['DATE'], item['AUTHOR'], item['CONTENT'], item['CLASS']])

    # A 'data' tömb elemei: ['dátum string', 'szerző', 'komment', 'osztály cimke ('0': nem spam, '1': spam)']

    # Train/test szétválasztás
    split = 0.7
    data = np.asarray(data)
    perm = np.random.permutation(len(data))

    train = data[perm][0:int(len(data) * split)]
    test = data[perm][int(len(data) * split):]

    print('Train set: ', np.shape(train))
    print('Test set: ', np.shape(test))

    return test


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

