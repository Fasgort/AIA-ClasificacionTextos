# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()

# np.linalg.norm(a-b) calcula la distancia euclidiana entre a = numpy.array y b = numpy.array

#def kmeans(num_clusters, data):

num_clusters = 2
data = iris.data

_shape = data.shape
clusters_centroid = np.empty([num_clusters, _shape[1]])
clusters = []
for c in range(num_clusters):
    clusters.append([])

# Inicialización de Forgy 
random_pick = np.arange(_shape[0])
np.random.shuffle(random_pick)
for c in range(num_clusters):
    clusters_centroid[c] = data[random_pick[c]]

## Inicialización de Partición Aleatoria
#min_data = np.min(data, axis=0)
#max_data = np.max(data, axis=0)
#for c in range(num_clusters):
#    randomValues = np.random.rand(_shape[1])
#    clusters_centroid[c] = randomValues * (max_data - min_data) + min_data

# Asignación a clústeres
for d in range(_shape[0]):
    closer_cluster = 0
    min_distance = np.linalg.norm(clusters_centroid[0]-data[d])
    for cluster in range(1, num_clusters):
        distance = np.linalg.norm(clusters_centroid[cluster]-data[d])
        if distance < min_distance:
            closer_cluster = cluster
            min_distance = distance
    clusters[closer_cluster].append(d)


    