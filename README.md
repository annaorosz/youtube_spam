# Youtube spam detector

# Author: Anna Orosz

### Dependencies: sci-kit learn, python 2.7

### Execution: %run main.py

To classify youtube comments as 'spam' or 'ham' (ham = not spam) I decided to tokenize and then analyze the comments and the
individual words using tf-idf.

This representation allowed me to build a sophisticated ML model using just "one" feature, the comments. Obviously,
to determine if a given comment was ham or spam, other features like the author and the date did not provide
useful information. When getting the frequencies of each word in each comment with regards to its classification
(0 = ham, 1 = spam), I was able to determine the nature of these comments with just the text.

Next, I needed to decide which classifier to use. First, I compared two classifiers: Multinomial Naive Bayes and a
Stochastic Gradient Descent Classifier. Later, I also added a third classifier to compare: an SVM (initially with a
cosine similarity kernel).

Upon using grid search, I have indeed found that it is most beneficial to use a cosine similarity kernel, with other
important observations in regards to the parameters. With this method, I was able to tune parameters such as the alpha,
the loss, etc in the case of the SGD. For both SGD and the SVM I was able to successfully tune the parameters
such that it improved the accuracy of the model by several percentage points.
