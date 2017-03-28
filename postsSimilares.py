# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
import unicodedata
import operator
import re
import random
from nltk import downloader
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
downloader.download("stopwords")
np.set_printoptions(threshold=50)

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
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)

# Preprocesamos los posts, siguiendo lo aprendido en la tarea 2
newsgroups_train.preprocessed = preprocess(newsgroups_train.data)

# Vectorizamos el conjunto de entrenamiento y test
vectorizer = TfidfVectorizer()
vectors_train = vectorizer.fit_transform(newsgroups_train.preprocessed)

# Generamos 6 clústeres a partir del conjunto de entrenamiento
# Un cluster por cada categoría
kmeans = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300).fit(vectors_train)
kmeans_train_index = kmeans.predict(vectors_train)

# Seleccionamos al azar, un post del conjunto de test, para emplear como consulta
post_num = random.randint(0, len(newsgroups_test.data))
post_data = newsgroups_test.data[post_num]
post_target = newsgroups_test.target[post_num]

# Preprocesamos la consulta
post_preprocessed = preprocess([post_data])[0]
vector_test = vectorizer.transform([post_preprocessed])[0]

# Consultamos a que clúster se aproxima más a la consulta
test_cluster = kmeans.predict(vector_test)

# Miramos que posts pertenecen al mismo clúster que la consulta
post_list = []
for p in range(len(kmeans_train_index)):
    if kmeans_train_index[p] == test_cluster:
        post_list.append(p)

# Realizamos medidas de similitud entre los posts del clúster y la consulta
relevancia = []
for d in range(len(post_list)):
    relevancia.append((post_list[d], newsgroups_train.target[post_list[d]], pairwise_distances(vectors_train.getrow(post_list[d]), vector_test, metric='cosine')[0][0]))
relevancia.sort(key=operator.itemgetter(2))

# Nos quedamos con los 5 posts más similares
similares = relevancia[:5:]

# Medimos cuantos de ellos pertenecen a la misma categoría
misma_categoria = 0
for s in similares:
    if s[1] == post_target:
        misma_categoria += 1

# Impresión de resultados
print()
print("Hemos cogido aleatoriamente un post del conjunto de test, para buscar aquellos posts más similares en el conjunto de entrenamiento.")
print("El nuevo post, empleado como consulta, pertenece a la categoría " + str(post_target) + " (" + str(newsgroups_test.target_names[post_target]) + ") y empieza así:")
print()
print("########################################################################################")
print(post_data[:300])
print("########################################################################################")
print()

print("Realizada la búsqueda de posts más similares, obtenemos los 5 siguientes resultados más similares.")
print("[INDEX, CATEGORIA, SIMILITUD_INVERSA]")
print()
print(similares)
print()

print("Podemos medir la eficacia de la clusterización, mediante el porcentaje de post similares que pertenecen a la misma categoría que el post de consulta.")
print("De 5 post similares, " + str(misma_categoria) + " pertenecen a la misma categoría, por lo que obtenemos un " + str(misma_categoria*100/5) + "% de precisión para este post en concreto.")
print()

print("El post más similar empieza así:")
print()
print("########################################################################################")
print(newsgroups_train.data[similares[0][0]][:300])
print("########################################################################################")
print()

# Clasificación
sum_score = 0
for post_num in range(len(newsgroups_test.data)):
    # Seleccionamos al azar, un post del conjunto de test, para emplear como consulta
    post_data = newsgroups_test.data[post_num]
    post_target = newsgroups_test.target[post_num]
    
    # Preprocesamos la consulta
    post_preprocessed = preprocess([post_data])[0]
    vector_test = vectorizer.transform([post_preprocessed])[0]
    
    # Consultamos a que clúster se aproxima más a la consulta
    test_cluster = kmeans.predict(vector_test)
    
    # Miramos que posts pertenecen al mismo clúster que la consulta
    post_list = []
    for p in range(len(kmeans_train_index)):
        if kmeans_train_index[p] == test_cluster:
            post_list.append(p)
    
    # Realizamos medidas de similitud entre los posts del clúster y la consulta
    relevancia = []
    for d in range(len(post_list)):
        relevancia.append((post_list[d], newsgroups_train.target[post_list[d]], pairwise_distances(vectors_train.getrow(post_list[d]), vector_test, metric='cosine')[0][0]))
    relevancia.sort(key=operator.itemgetter(2))
    
    # Nos quedamos con los 5 posts más similares
    similares = relevancia[:5:]
    
    # Medimos cuantos de ellos pertenecen a la misma categoría
    misma_categoria = 0
    for s in similares:
        if s[1] == post_target:
            misma_categoria += 1
    sum_score += misma_categoria/5
    
    if (post_num+1) % 20 == 0 and post_num != 0:
        print("Se han evaluado " + str(post_num+1) + " posts del conjunto de test, de " + str(len(newsgroups_test.data)) + " totales.")
        print("Precisión provisional de la clasificación: " + str(sum_score/(post_num+1)))
        print()

# Si el usuario ha sido suficientemente paciente, presentamos la precisión final
sum_score /= len(newsgroups_test.data)
print()
print("########################################################################################")
print("Precisión final de la clusterización: " + str(sum_score))
print("########################################################################################")
