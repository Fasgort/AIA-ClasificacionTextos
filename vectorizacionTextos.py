# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
from nltk import downloader
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
downloader.download("stopwords")

stopwords_list = set(stopwords.words("spanish"))
stemmer = SnowballStemmer("spanish")

documentos = []
documentos.append("oro plata camión") # Esta es la consulta
documentos.append("Éste texto no tiene nada que ver con los demás")
documentos.append("La plata fue entregada en camiones color plata")
documentos.append("El cargamento de oro llegó en un camión. El cargamento de oro llegó en un camión. El cargamento de oro llegó en un camión")
documentos.append("Cargamentos de oro dañados por el fuego")
documentos.append("El cargamento de oro llegó en un camión")

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

