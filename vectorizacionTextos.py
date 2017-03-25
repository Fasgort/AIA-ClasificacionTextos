# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import unicodedata
import operator
from nltk import downloader
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
downloader.download("stopwords")
np.set_printoptions(threshold=np.nan)

stopwords_list = set(stopwords.words("spanish"))
stemmer = SnowballStemmer("spanish")

def tratamiento1(documentos):
    # Tratamiento de datos básico
    new_documentos = []
    for d in range(len(documentos)):
        unaccented_text = ''.join(c for c in unicodedata.normalize('NFD', documentos[d]) if unicodedata.category(c) != 'Mn')
        lower_words = [str.lower(word) for word in unaccented_text.split(" ")]
        new_documentos.append(" ".join(lower_words))
    return new_documentos

def tratamiento2(documentos):
    # Tratamiento de datos básico + stopwords
    new_documentos = []
    for d in range(len(documentos)):
        unaccented_text = ''.join(c for c in unicodedata.normalize('NFD', documentos[d]) if unicodedata.category(c) != 'Mn')
        lower_words = [str.lower(word) for word in unaccented_text.split(" ")]
        filtered_words = [word for word in lower_words if word not in stopwords_list]
        new_documentos.append(" ".join(filtered_words))
    return new_documentos

def tratamiento3(documentos):
    # Tratamiento de datos básico + stopwords + stemming
    new_documentos = []
    for d in range(len(documentos)):
        unaccented_text = ''.join(c for c in unicodedata.normalize('NFD', documentos[d]) if unicodedata.category(c) != 'Mn')
        lower_words = [str.lower(word) for word in unaccented_text.split(" ")]
        filtered_words = [word for word in lower_words if word not in stopwords_list]
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        new_documentos.append(" ".join(stemmed_words))
    return new_documentos

# Main

documentosOrig = []
documentosOrig.append("oro plata camión") # Esta es la consulta
documentosOrig.append("Éste texto no tiene nada que ver con los demás")
documentosOrig.append("La plata fue entregada en camiones color plata")
documentosOrig.append("El cargamento de oro llegó en un camión. El cargamento de oro llegó en un camión. El cargamento de oro llegó en un camión")
documentosOrig.append("Cargamentos de oro dañados por el fuego")
documentosOrig.append("El cargamento de oro llegó en un camión")

print()
print("////////////////////////////////")
print("///////////RESULTADOS///////////")
print("////////////////////////////////")
print()
print("Menor distancia equivale a mayor relevancia en la consulta.")
print()

###
### EJERCICIO 1
###
documentos = tratamiento1(documentosOrig)
countVectorizer = CountVectorizer()
documentosVect = countVectorizer.fit_transform(documentos[1::]) # Ignoramos la consulta en la generación del vectorizador
consultaVect = countVectorizer.transform(documentos[:1:]) # Generamos el vector de la consulta
                                        
print("Con el modelo vectorial simple, y la distancia euclídea")
relevanciaDocEj1 = []
for d in range(documentosVect.shape[0]):
    relevanciaDocEj1.append((d+1,pairwise_distances(documentosVect.getrow(d), consultaVect, metric='euclidean')[0][0]))
relevanciaDocEj1.sort(key=operator.itemgetter(1))
print(relevanciaDocEj1)
print()

###
### EJERCICIO 2
### Comentario: La distancia devuelta es 1 - similitud coseno
### por lo que los resultados más relevantes son los que se acercan a 0.
###
print("Con el modelo vectorial simple, y la distancia normalizada (coseno)")
relevanciaDocEj2 = []
for d in range(documentosVect.shape[0]):
    relevanciaDocEj2.append((d+1,pairwise_distances(documentosVect.getrow(d), consultaVect, metric='cosine')[0][0]))
relevanciaDocEj2.sort(key=operator.itemgetter(1))
print(relevanciaDocEj2)
print()

###
### EJERCICIO 3
###
documentos = tratamiento2(documentosOrig)
countVectorizer = CountVectorizer()
documentosVect = countVectorizer.fit_transform(documentos[1::]) # Ignoramos la consulta en la generación del vectorizador
consultaVect = countVectorizer.transform(documentos[:1:]) # Generamos el vector de la consulta

print("Con el modelo vectorial simple, la distancia normalizada (coseno) y empleando stopwords")
relevanciaDocEj3 = []
for d in range(documentosVect.shape[0]):
    relevanciaDocEj3.append((d+1,pairwise_distances(documentosVect.getrow(d), consultaVect, metric='cosine')[0][0]))
relevanciaDocEj3.sort(key=operator.itemgetter(1))
print(relevanciaDocEj3)
print()

###
### EJERCICIO 4
###
documentos = tratamiento3(documentosOrig)
countVectorizer = CountVectorizer()
documentosVect = countVectorizer.fit_transform(documentos[1::]) # Ignoramos la consulta en la generación del vectorizador
consultaVect = countVectorizer.transform(documentos[:1:]) # Generamos el vector de la consulta

print("Con el modelo vectorial simple, la distancia normalizada (coseno), empleando stopwords y stemming")
relevanciaDocEj4 = []
for d in range(documentosVect.shape[0]):
    relevanciaDocEj4.append((d+1,pairwise_distances(documentosVect.getrow(d), consultaVect, metric='cosine')[0][0]))
relevanciaDocEj4.sort(key=operator.itemgetter(1))
print(relevanciaDocEj4)
print()

###
### EJERCICIO 5
###
documentos = tratamiento3(documentosOrig)
tfidVectorizer = TfidfVectorizer()
documentosVect = tfidVectorizer.fit_transform(documentos[1::]) # Ignoramos la consulta en la generación del vectorizador
consultaVect = tfidVectorizer.transform(documentos[:1:]) # Generamos el vector de la consulta

print("Con el modelo de frecuencia documental inversa, la distancia normalizada (coseno), empleando stopwords y stemming")
relevanciaDocEj5 = []
for d in range(documentosVect.shape[0]):
    relevanciaDocEj5.append((d+1,pairwise_distances(documentosVect.getrow(d), consultaVect, metric='cosine')[0][0]))
relevanciaDocEj5.sort(key=operator.itemgetter(1))
print(relevanciaDocEj5)
print()