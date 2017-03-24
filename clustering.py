# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()

# np.linalg.norm(a-b) calcula la distancia euclidiana entre a = numpy.array y b = numpy.array

#def kmeans(num_clusters, data):

num_clusters = 2
data = iris.data

_shape = data.shape
clusters = np.empty([num_clusters, _shape[1]])

# Inicialización de Forgy 
random_pick = np.arange(_shape[0])
np.random.shuffle(random_pick)
for c in range(num_clusters):
    clusters[c] = data[random_pick[c]]

# Inicialización de Partición Aleatoria
min_data = np.min(data, axis=0)
max_data = np.max(data, axis=0)
for c in range(num_clusters):
    randomValues = np.random.rand(_shape[1])
    clusters[c] = randomValues * (max_data - min_data) + min_data
