# -*- coding: utf-8 -*-

import json
import re
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve


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
    
    # Entrena y clasifica
    gs.fit(X_train, Y_train)
    pred = gs.predict(X_test)
                 
    ### Medidas de rendimiento
    accuracy = accuracy_score(Y_test, pred)
    precision = precision_score(Y_test, pred)
    recall = recall_score(Y_test, pred)
    f_score = f1_score(Y_test, pred)
    fpr, tpr, _ = roc_curve(Y_test, pred)
    auc_score = auc(fpr, tpr)
    precision_recall = [[],[]]
    precision_recall[0], precision_recall[1], _ = precision_recall_curve(Y_test, pred)
    
    return accuracy, precision, recall, f_score, auc_score, precision_recall, gs.best_params_

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
    'mnb_clf__alpha': [0.1, 1, 10]
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
        tweets1.append([1, tweet[1]])
        tweets2.append([1, tweet[1]])
        tweets3.append([1, tweet[1]])
        tweets4.append([0, tweet[1]])
    if tweet[0] == 'negative':
        tweets1.append([0, tweet[1]])
        tweets2.append([1, tweet[1]])
        tweets3.append([0, tweet[1]])
        tweets4.append([1, tweet[1]])
    if tweet[0] == 'neutral':
        tweets2.append([0, tweet[1]])
        tweets3.append([0, tweet[1]])
        tweets4.append([0, tweet[1]])
    if tweet[0] == 'irrelevant':
        tweets3.append([0, tweet[1]])
        tweets4.append([0, tweet[1]])

### Clasificación de los 4 problemas
# Problema 1
accuracy, precision, recall, f_score, auc_score, precision_recall, best_params = clasifica(gs, tweets1)
print("##################################################")
print()
print("Para el problema de clasificación \"Positivos vs negativos\", obtenemos los siguientes resultados:")
print()
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F-score: " + str(f_score))
print("AUC score: " + str(auc_score))
print()
plt.clf()
plt.plot(precision_recall[0], precision_recall[1], lw=2, color='navy')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')
plt.show()
print()
print("Los mejores parámetros para el pipeline fueron los siguientes:")
print()
print("ngram_range: " + str(best_params["tfidf_vect__ngram_range"]))
print("stop_words: " + str(best_params["tfidf_vect__stop_words"]))
print("smooth_idf: " + str(best_params["tfidf_vect__smooth_idf"]))
print("use_idf: " + str(best_params["tfidf_vect__use_idf"]))
print("sublinear_tf: " + str(best_params["tfidf_vect__sublinear_tf"]))
print("binary: " + str(best_params["tfidf_vect__binary"]))
print("alpha: " + str(best_params["mnb_clf__alpha"]))
print()
print("##################################################")
print()
# Resultados
#
# Para el problema de clasificación "Positivos vs negativos", obtenemos los siguientes resultados:
#
# Accuracy: 0.771929824561
# Precision: 0.826086956522
# Recall: 0.678571428571
# F-score: 0.745098039216
# AUC score: 0.770320197044
#
# (Imagen con la curva Precision/Recall)
#
# Los mejores parámetros para el pipeline fueron los siguientes:
#
# ngram_range: (1, 1)
# stop_words: None
# smooth_idf: True
# use_idf: True
# sublinear_tf: True
# binary: False
# alpha: 0.1
#
###

# Problema 2
accuracy, precision, recall, f_score, auc_score, precision_recall, best_params = clasifica(gs, tweets2)
print("##################################################")
print()
print("Para el problema de clasificación \"Con sentimiento vs sin sentimiento\", obtenemos los siguientes resultados:")
print()
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F-score: " + str(f_score))
print("AUC score: " + str(auc_score))
print()
plt.clf()
plt.plot(precision_recall[0], precision_recall[1], lw=2, color='navy')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')
plt.show()
print()
print("Los mejores parámetros para el pipeline fueron los siguientes:")
print()
print("ngram_range: " + str(best_params["tfidf_vect__ngram_range"]))
print("stop_words: " + str(best_params["tfidf_vect__stop_words"]))
print("smooth_idf: " + str(best_params["tfidf_vect__smooth_idf"]))
print("use_idf: " + str(best_params["tfidf_vect__use_idf"]))
print("sublinear_tf: " + str(best_params["tfidf_vect__sublinear_tf"]))
print("binary: " + str(best_params["tfidf_vect__binary"]))
print("alpha: " + str(best_params["mnb_clf__alpha"]))
print()
print("##################################################")
print()
# Resultados
#
# Para el problema de clasificación "Con sentimiento vs sin sentimiento", obtenemos los siguientes resultados:
#
# Accuracy: 0.779661016949
# Precision: 0.716666666667
# Recall: 0.413461538462
# F-score: 0.524390243902
# AUC score: 0.672730769231
#
# (Imagen con la curva Precision/Recall)
#
# Los mejores parámetros para el pipeline fueron los siguientes:
#
# ngram_range: (1, 2)
# stop_words: None
# smooth_idf: True
# use_idf: False
# sublinear_tf: True
# binary: False
# alpha: 0.1
#
###

# Problema 3
accuracy, precision, recall, f_score, auc_score, precision_recall, best_params = clasifica(gs, tweets3)
print("##################################################")
print()
print("Para el problema de clasificación \"Positivos vs no positivos\", obtenemos los siguientes resultados:")
print()
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F-score: " + str(f_score))
print("AUC score: " + str(auc_score))
print()
plt.clf()
plt.plot(precision_recall[0], precision_recall[1], lw=2, color='navy')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')
plt.show()
print()
print("Los mejores parámetros para el pipeline fueron los siguientes:")
print()
print("ngram_range: " + str(best_params["tfidf_vect__ngram_range"]))
print("stop_words: " + str(best_params["tfidf_vect__stop_words"]))
print("smooth_idf: " + str(best_params["tfidf_vect__smooth_idf"]))
print("use_idf: " + str(best_params["tfidf_vect__use_idf"]))
print("sublinear_tf: " + str(best_params["tfidf_vect__sublinear_tf"]))
print("binary: " + str(best_params["tfidf_vect__binary"]))
print("alpha: " + str(best_params["mnb_clf__alpha"]))
print()
print("##################################################")
print()
# Resultados
#
# Para el problema de clasificación "Positivos vs no positivos", obtenemos los siguientes resultados:
#
# Accuracy: 0.874666666667
# Precision: 0.75
# Recall: 0.117647058824
# F-score: 0.203389830508
# AUC score: 0.555737109659
#
# (Imagen con la curva Precision/Recall)
#
# Los mejores parámetros para el pipeline fueron los siguientes:
#
# ngram_range: (1, 2)
# stop_words: None
# smooth_idf: True
# use_idf: False
# sublinear_tf: True
# binary: True
# alpha: 0.1
#
###

# Problema 4
accuracy, precision, recall, f_score, auc_score, precision_recall, best_params = clasifica(gs, tweets4)
print("##################################################")
print()
print("Para el problema de clasificación \"Negativos vs no negativos\", obtenemos los siguientes resultados:")
print()
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F-score: " + str(f_score))
print("AUC score: " + str(auc_score))
print()
plt.clf()
plt.plot(precision_recall[0], precision_recall[1], lw=2, color='navy')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')
plt.show()
print()
print("Los mejores parámetros para el pipeline fueron los siguientes:")
print()
print("ngram_range: " + str(best_params["tfidf_vect__ngram_range"]))
print("stop_words: " + str(best_params["tfidf_vect__stop_words"]))
print("smooth_idf: " + str(best_params["tfidf_vect__smooth_idf"]))
print("use_idf: " + str(best_params["tfidf_vect__use_idf"]))
print("sublinear_tf: " + str(best_params["tfidf_vect__sublinear_tf"]))
print("binary: " + str(best_params["tfidf_vect__binary"]))
print("alpha: " + str(best_params["mnb_clf__alpha"]))
print()
print("##################################################")
print()
# Resultados
#
# Para el problema de clasificación "Negativos vs no negativos", obtenemos los siguientes resultados:
#
# Accuracy: 0.869333333333
# Precision: 0.857142857143
# Recall: 0.28125
# F-score: 0.423529411765
# AUC score: 0.635801848875
#
# (Imagen con la curva Precision/Recall)
#
# Los mejores parámetros para el pipeline fueron los siguientes:
#
# ngram_range: (1, 1)
# stop_words: None
# smooth_idf: False
# use_idf: True
# sublinear_tf: False
# binary: False
# alpha: 0.1
#
###

