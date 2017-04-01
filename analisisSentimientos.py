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


             
