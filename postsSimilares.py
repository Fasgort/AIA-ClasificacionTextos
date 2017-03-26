# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
import unicodedata
import re
import random
from nltk import downloader
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
downloader.download("stopwords")
np.set_printoptions(threshold=np.nan)

def preprocess(documentos):
    # Tratamiento de datos básico + stopwords + stemming
    new_documentos = []
    stopwords_list = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")
    for d in range(len(documentos)):
        unaccented_text = ''.join(c for c in unicodedata.normalize('NFD', documentos[d]) if unicodedata.category(c) != 'Mn')
        clean_words = re.findall(r'(?ms)\W*(\w+)', unaccented_text) # Útil para eliminar símbolos y similares
        lower_words = [str.lower(word) for word in clean_words]
        filtered_words = [word for word in lower_words if word not in stopwords_list]
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        new_documentos.append(" ".join(stemmed_words))
    return new_documentos

# Seleccionamos las categorías a emplear para el ejercicio
categories = ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x", "sci.space"]

# Cargamos los conjuntos de entrenamiento y test
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
newsgroup_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)

# Preprocesamos los posts, siguiendo lo aprendido en la tarea 2
newsgroups_train.data = preprocess(newsgroups_train.data)

# Vectorizamos el conjunto de entrenamiento
vectorizer = TfidfVectorizer()
vectors_train = vectorizer.fit_transform(newsgroups_train.data)

# Generamos 6 clústeres a partir del conjunto de entrenamiento
kmeans = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300).fit(vectors_train)

# Seleccionamos al azar, un post del conjunto de test, para emplear como consulta
post_num = random.randint(0, len(newsgroup_test.data))
