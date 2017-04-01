# -*- coding: utf-8 -*-

import json
import re
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc



### Carga del corpus
tweets = [] # classification, tweet
f = open('AnalisisSentimientos\\corpus.csv', 'r')
for line in f:
    line = re.sub('["\\n]', '', line)
    tweet = line.split(",")
    tweet = [tweet[1], tweet[2]]
    try:
        jsonf = open('AnalisisSentimientos\\rawdata\\' + str(tweet[1]) + '.json' )
        json_data = json.load(jsonf)
        if json_data["lang"] != "en":
            continue
        tweet[1] = json_data["text"]
    except:
        continue
    tweets.append(tweet)

### Cambio del target para diferentes problemas de clasificación
for i in range(len(tweets)):
    tweet = tweets[i]
    if tweet[0] == 'positive' or tweet[0] == 'negative':
        tweet[0] = 'con_sentimiento'
    else:
        tweet[0] = 'sin_sentimiento'
        
### Creación de los conjuntos de entrenamiento y test
shuffle(tweets)
splitpoint = int((len(tweets)/5)*4)
tweets_train = tweets[:splitpoint]
tweets_test = tweets[splitpoint:]

### Creación del pipeline
estimators = [('tfidf_vect', TfidfVectorizer()), ('mnb_clf', MultinomialNB())]
pipeline = Pipeline(estimators)

### Búsqueda de los parámetros óptimos
parameters = {
    'tfidf_vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf_vect__stop_words': ["english", None],
    'tfidf_vect__smooth_idf': [True, False],
    'tfidf_vect__use_idf': [True, False],
    'tfidf_vect__sublinear_tf': [True, False],
    'tfidf_vect__binary': [True, False],
    'tfidf_vect__max_df':[0.5],
    'mnb_clf__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    }

gs = GridSearchCV(pipeline, parameters, cv = ShuffleSplit(n_splits=3, test_size=0.2))

X_train = [x[1] for x in tweets_train]
Y_train = [x[0] for x in tweets_train]
X_test = [x[1] for x in tweets_test]
Y_test = [x[0] for x in tweets_train]

gs.fit(X_train, Y_train)
pred = gs.predict(X_test)
             
