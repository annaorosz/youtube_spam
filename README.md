# Youtube spam comments detector

## Author: Anna Orosz

### Dependencies: sci-kit learn, python 2.7
### Execution: run main.py

To classify youtube comments as 'spam' or 'ham' (ham = not spam) I used tf-idf (got the frequency of each word
across the whole dataset) and then built 2 different models with the transformed data.

The tf-idf representation allowed me to build a sophisticated ML model using just "one" feature, the content,
which consisted of the array of comments in each instance.
Obviously, to determine if a given comment was ham or spam, other features like the author and the date did not provide
useful information. When getting the frequencies of each word in each comment with regards to its classification
(0 = ham, 1 = spam), I was able to determine the nature of these comments using text classification.

Next, I needed to decide which classifier to use. First, I compared two classifiers: Multinomial Naive Bayes and a
Stochastic Gradient Descent Classifier with a 'hinge' loss, which works as a Support Vector Machine.
Later, I also added a third classifier to compare: an SVM with a cosine similarity kernel. Upon inspection I have found
that the SVM with the cosine similarity kernel performed slightly better than the SGD with hinge loss, so I
removed the SGD model and decided to compare the Multinomial NB model with the SVM model.

Upon using grid search, I have found the best possible MNB as well as SVM by tuning the parameters.
By setting these parameters, both models were able to achieve ~95% or higher recall rate, which is an important
metric to measure the performance of the model.

This metric divides the true positive values found by all positive values found.
because it is more important to find the 'ham' comments than the spam comments. This is because it is less of
a problem to classify a spam as ham and have an additional comment than the other way around, which would lead
to disregarding the comment altogether.
Another important metric which is widely used in the field of Natural Language processing, which, upon inspection
I have found to be the best measurement of performance for this problem. This harmonic mean takes sort of an average
between recall and precision (another useful text classification metric). However, when the two values differ by
a lot, it tends to be biased towards the lower value.


