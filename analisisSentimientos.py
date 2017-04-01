# -*- coding: utf-8 -*-

import json
import re
from copy import deepcopy
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


### Método que entrena el pipeline, clasifica y devuelve medidas de rendimiento
def clasifica(gs, tweets):

    ### Creación de los conjuntos de entrenamiento y test
    shuffle(tweets)
    splitpoint = int((len(tweets)/5)*4)
    tweets_train = tweets[:splitpoint]
    tweets_test = tweets[splitpoint:]
    
    X_train = [x[1] for x in tweets_train]
    Y_train = [x[0] for x in tweets_train]
    X_test = [x[1] for x in tweets_test]
    Y_test = [x[0] for x in tweets_test]
    
    gs.fit(X_train, Y_train)
    pred = gs.predict(X_test)
                 
    ### Medidas de rendimiento
    accuracy = accuracy_score(Y_test, pred)
    precision = precision_score(Y_test, pred)
    recall = recall_score(Y_test, pred)
    f_score = f1_score(Y_test, pred)
    fpr, tpr, _ = roc_curve(Y_test, pred, pos_label=2)
    auc_ratio = auc(fpr, tpr)
    
    return accuracy, precision, recall, f_score, auc_ratio

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

### Creación del pipeline
estimators = [('tfidf_vect', TfidfVectorizer()), ('mnb_clf', MultinomialNB())]
pipeline = Pipeline(estimators)

### Implementación de GridSearchCV
parameters = {
    'tfidf_vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf_vect__stop_words': ["english", None],
    'tfidf_vect__smooth_idf': [True, False],
    'tfidf_vect__use_idf': [True, False],
    'tfidf_vect__sublinear_tf': [True, False],
    'tfidf_vect__binary': [True, False],
    'tfidf_vect__max_df': [0.5],
    'mnb_clf__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    }

gs = GridSearchCV(pipeline, parameters, cv = ShuffleSplit(n_splits=3, test_size=0.2))

### Creación de diferentes conjuntos de datos para los 4 problemas de clasificación
tweets1 = [] # Positivos vs negativos
tweets2 = [] # Con sentimiento vs sin sentimiento
tweets3 = [] # Positivos vs el resto
tweets4 = [] # Negativos vs el resto
for t in range(len(tweets)):
    tweet = tweets[t]
    if tweet[0] == 'positive':
        tweets1.append(tweet)
        tweets2.append(["con_sentimiento", tweet[1]])
        tweets3.append(tweet)
        tweets4.append(["no_negativo", tweet[1]])
    if tweet[0] == 'negative':
        tweets1.append(tweet)
        tweets2.append(["con_sentimiento", tweet[1]])
        tweets3.append(["no_positivo", tweet[1]])
        tweets4.append(tweet)
    if tweet[0] == 'neutral':
        tweets2.append(["sin_sentimiento", tweet[1]])
        tweets3.append(["no_positivo", tweet[1]])
        tweets4.append(["no_negativo", tweet[1]])
    if tweet[0] == 'irrelevant':
        tweets3.append(["no_positivo", tweet[1]])
        tweets4.append(["no_negativo", tweet[1]])

### Clasificación de los 4 problemas
# Problema 1
accuracy, precision, recall, f_score, auc_ratio = clasifica(gs, tweets1)
print("Para el problema de clasificación \"Positivos vs negativos\", obtenemos los siguientes resultados:")
print()
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F-score: " + str(f_score))
print("AUC ratio: " + str(auc_ratio))
print()
print()

# Problema 2
accuracy, precision, recall, f_score, auc_ratio = clasifica(gs, tweets2)
print("Para el problema de clasificación \"Con sentimiento vs sin sentimiento\", obtenemos los siguientes resultados:")
print()
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F-score: " + str(f_score))
print("AUC ratio: " + str(auc_ratio))
print()
print()

# Problema 3
accuracy, precision, recall, f_score, auc_ratio = clasifica(gs, tweets3)
print("Para el problema de clasificación \"Positivos vs no positivos\", obtenemos los siguientes resultados:")
print()
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F-score: " + str(f_score))
print("AUC ratio: " + str(auc_ratio))
print()
print()

# Problema 4
accuracy, precision, recall, f_score, auc_ratio = clasifica(gs, tweets4)
print("Para el problema de clasificación \"Negativos vs no negativos\", obtenemos los siguientes resultados:")
print()
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F-score: " + str(f_score))
print("AUC ratio: " + str(auc_ratio))
print()
print()

