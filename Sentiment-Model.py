import sklearn
import re
import pandas as pd
import numpy
import nltk
from nltk.corpus import stopwords
import string
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.svm import SVC
import scikitplot as skplt
from sklearn.metrics import auc, roc_curve
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
import matplotlib.pyplot as plt

#This program loads Amazon reviews and their corresponding star values (1-5) from a csv file
#The reviews are processed to remove non alphabetic characters, stop words, and punctuation
#Unigram, Bigram, and Trigram features are then extracted from the reviews
#The reviews are vectorized (Count or TFIDF vectorization) and split into training and testing sets
# Using different Machine Learning algorithms (Naive Bayes, KNN, SGD, SVC), the model predicts the star value of a given text review
#The predicted star values are then compared to the actual star reviews of the testing set (supervised learning)"""


# Loads data from csv and establishes different clases for positive and negative reviews
train_data = pd.read_csv("C:/Users/conno/Downloads/amazon_reviews.csv")
train_class = train_data[(train_data['Score'] == 1) | (train_data['Score'] == 5)]


#Casts the text reviews and their star values as two different classes
review_class = train_class['Review']
score_class = train_class['Score']

def clean_text(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """
    s = re.sub("[^a-z A-Z]", "", s)
    s = s.replace(' n ', ' ')
    return s.lower()

def text_process(text):
    """Re-formats the punctuation used in review. Removes 'stopwords' E.g (the, and, but, or)"""
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [clean_text(word) for word in nopunc.split()]

def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    """Splits each review into a list of bigrams E.g "The book is good" - (the book), (book is), (is good).
     Filters out the top 200 most relevant bigrams with a chi-squared association measure"""
    words=text_process(words)
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return ([ngram for ngram in bigrams]+ [(word,) for word in words])
    #return ([ngram for ngram in words, bigrams])

def trigram_word_feats(words, score_fn=TrigramAssocMeasures.chi_sq, n=100):
    """Splits each review into a list of trigrams E.g "The book is good" - (the book is), (book is good)
    Filters out the top 100 most relevant trigrams with a chi-squared association measure """
    words=text_process(words)
    trigram_finder = TrigramCollocationFinder.from_words(words)
    trigrams = trigram_finder.nbest(score_fn, n)
    bigrams= bigram_word_feats(review_class)
    return ([ngram for ngram in trigrams]+ bigrams)



"Naive Bayes Count Vectorizer"

#Vectorizes the words of a review into a count matrix
bow_transformer = sklearn.feature_extraction.text.CountVectorizer(analyzer=text_process).fit(review_class)

#Applies the bow_transformer to the reviews
X_review = bow_transformer.transform(review_class)

#Splits the reviews and ratings into training and testing classes (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_review, score_class, test_size=0.3, random_state=101)

# Fits a Naive Bayes model to the training reviews and scores
nb = MultinomialNB()
nb.fit(X_train, y_train)

#Predicts the scores from the written testing revies
preds = nb.predict(X_test)

#Comparing how well the model performed against the actually review scores
print(accuracy_score(y_test,preds))
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))


"Naive Bayes Tf-idf vectorizer"

review_class = train_class['Review']
score_class = train_class['Score']

#Vectorizes the words of a review into a TF_IDF matrix
tf_transformer = sklearn.feature_extraction.text.TfidfVectorizer(analyzer=text_process).fit(review_class)

#Applies the TF_IDF vectorizer to the words in each review
X_review = tf_transformer.transform(review_class)

#Splits the reviews and ratings into training and testing classes (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_review, score_class, test_size=0.3, random_state=101)

# Fits a Naive Bayes model to the training reviews and scores
nb = MultinomialNB()
nb.fit(X_train, y_train)

#Predicts the scores from the written testing revies
preds = nb.predict(X_test)
prob = nb.predict_proba(X_test)

#Comparing how well the model performed against the actually review scores
print(accuracy_score(y_test,preds))
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))

"Naive Bayes Count Vectorizer (with top 20% Bigrams)"

#Vectorizes the words of a review into a count matrix (includes bigram word features)
bow_transformer = sklearn.feature_extraction.text.CountVectorizer(analyzer=bigram_word_feats).fit(review_class)

#Applies the bow_transformer to the reviews
X_review = bow_transformer.transform(review_class)

#Splits the reviews and ratings into training and testing classes (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_review, score_class, test_size=0.3, random_state=101)

# Fits a Naive Bayes model to the training reviews and scores
nb = MultinomialNB()
nb.fit(X_train, y_train)

#Predicts the scores from the written testing revies
preds = nb.predict(X_test)

#Comparing how well the model performed against the actually review scores
print(accuracy_score(y_test,preds))
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))

"Naive Bayes Count Vectorizer (with top 20% Bigrams and top 10% trigrams)"

#Vectorizes the words of a review into a count matrix (includes Bigram and Trigram word features)
bow_transformer = sklearn.feature_extraction.text.CountVectorizer(analyzer=trigram_word_feats).fit(review_class)

#Applies the bow_transformer to the reviews
X_review = bow_transformer.transform(review_class)

#Splits the reviews and ratings into training and testing classes (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_review, score_class, test_size=0.3, random_state=101)

# Fits a Naive Bayes model to the training reviews and scores
nb = MultinomialNB()
nb.fit(X_train, y_train)

#Predicts the scores from the written testing revies
preds = nb.predict(X_test)

#Comparing how well the model performed against the actually review scores
print(accuracy_score(y_test,preds))
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))


"SVC Tf-idf Vectorizer"

X = train_class['Long']
y = train_class['Rating']

vectorizer = TfidfVectorizer(analyzer=text_process).fit(X)
X= vectorizer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

sv= LinearSVC()
sv.fit(X_train, y_train)
preds = sv.predict(X_test)

print(accuracy_score(y_test,preds))
print('\n')

print(classification_report(y_test, preds))

"SVC Count Vectorizer"

X = train_class['Long']
y = train_class['Rating']

bow_transformer = sklearn.feature_extraction.text.CountVectorizer(analyzer=text_process).fit(X)

X = bow_transformer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

sv= LinearSVC()
sv.fit(X_train, y_train)
preds = sv.predict(X_test)

print(accuracy_score(y_test,preds))
print('\n')
print(classification_report(y_test, preds))

"KNN Count Vectorizer"

bow_transformer = sklearn.feature_extraction.text.CountVectorizer(analyzer=text_process).fit(X)

X = bow_transformer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


KN = KNeighborsClassifier(n_neighbors=15)
KN.fit(X_train, y_train)

preds = KN.predict(X_test)

print(accuracy_score(y_test,preds))
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))

"KNN Tf-idf Vectorizer"

X = train_class['Long']
y = train_class['Rating']

tf_transformer = sklearn.feature_extraction.text.TfidfVectorizer(analyzer=text_process).fit(X)

X = tf_transformer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

KN = KNeighborsClassifier(n_neighbors=15)
KN.fit(X_train, y_train)

preds = KN.predict(X_test)

print(accuracy_score(y_test,preds))
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))

"SGD Tf-idf Vectorizer"

vectorizer = TfidfVectorizer(analyzer=text_process).fit(X)
X= vectorizer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

sgd= SGDClassifier()
sgd.fit(X_train, y_train)
preds = sgd.predict(X_test)

print(accuracy_score(y_test,preds))
print('\n')

print(classification_report(y_test, preds))

"SGD Count Vectorizer"

X = train_class['Long']
y = train_class['Rating']

bow_transformer = sklearn.feature_extraction.text.CountVectorizer(analyzer=text_process).fit(X)

X = bow_transformer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

sgd= SGDClassifier()
sgd.fit(X_train, y_train)
preds = sgd.predict(X_test)

print(accuracy_score(y_test,preds))
print('\n')
print(classification_report(y_test, preds))






