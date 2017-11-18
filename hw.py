
# coding: utf-8

# In[1]:

import numpy as np
import random
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics


# In[2]:

data = []
X = []
y = []

# Betöltés
with open('data.csv','r') as f:
    csvreader = csv.DictReader(f)
    for item in csvreader:
        # A 'data' tömb elemei: ['dátum string', 'szerző', 'komment', 'osztály cimke ('0': nem spam, '1': spam)']
        data.append([ item['DATE'], item['AUTHOR'], item['CONTENT'], item['CLASS'] ])
        
        X.append(item['CONTENT'])
        y.append(item['CLASS'])


# In[3]:

# Train/test szétválasztás
split = 0.7
data = np.asarray(X)
perm = np.random.permutation(len(X))
labels = np.asarray(y)
perm2 = np.random.permutation(len(y))

X_train = data[perm][0:int(len(data)*split)]
X_test = data[perm][int(len(data)*split):]

y_train = labels[perm][0:int(len(labels)*split)]
y_test = labels[perm][int(len(labels)*split):]

print('X Train set: ', np.shape(X_train))
print('X Test set: ', np.shape(X_test))

print('y Train set: ', np.shape(y_train))
print('y Test set: ', np.shape(y_test))


# In[4]:

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape


# In[5]:

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[6]:

clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[7]:

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf.fit(X_train, y_train) 


# In[8]:

predicted = text_clf.predict(X_test)
np.mean(predicted == y_test) 
print(metrics.classification_report(y_test, predicted))


# In[21]:

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])
text_clf.fit(X_train, y_train) 


# In[22]:

predicted = text_clf.predict(X_test)
print (np.mean(predicted == y_test))
print(metrics.classification_report(y_test, predicted))


# In[14]:



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
sum_positive = 0
found_positive = 0

for datapoint in test:
    if datapoint[-1] == '1':
        sum_positive += 1
        if dumb_classify(datapoint) == '1':
            found_positive += 1
    
print('Recall:', found_positive / sum_positive)


