# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()

num_clusters = 3
data = iris.data
initial_method = 0 # 0 - Forgy; 1 - Partición Aleatoria

def kmeans(num_clusters, data, initial_method):
    # Implementación k-medios
    # Función de distancia: Distancia euclidiana
    # Emplea todos los atributos de data en el cálculo de la distancia
    
    _shape = data.shape
    clusters_centroid = np.empty([num_clusters, _shape[1]])
    clusters = []
    for c in range(num_clusters):
        clusters.append([])
    
    if initial_method == 0:
        # Inicialización de Forgy 
        random_pick = np.arange(_shape[0])
        np.random.shuffle(random_pick)
        for c in range(num_clusters):
            clusters_centroid[c] = data[random_pick[c]]
    elif initial_method == 1:
        ## Inicialización de Partición Aleatoria
        min_data = np.min(data, axis=0)
        max_data = np.max(data, axis=0)
        for c in range(num_clusters):
            randomValues = np.random.rand(_shape[1])
            clusters_centroid[c] = randomValues * (max_data - min_data) + min_data
    
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
    
    # Bucle
    while True:
        
        new_clusters = []
        for c in range(num_clusters):
            new_clusters.append([])
        
        # Cálculo de los nuevos centroides
        for c in range(num_clusters):
            sum = np.zeros(_shape[1])
            for d in range(len(clusters[c])):
                sum += data[clusters[c][d]]
            clusters_centroid[c] = sum / len(clusters[c])
            
        # Asignación a clústeres
        for d in range(_shape[0]):
            closer_cluster = 0
            min_distance = np.linalg.norm(clusters_centroid[0]-data[d])
            for cluster in range(1, num_clusters):
                distance = np.linalg.norm(clusters_centroid[cluster]-data[d])
                if distance < min_distance:
                    closer_cluster = cluster
                    min_distance = distance
            new_clusters[closer_cluster].append(d)
            
        condition = False
        for c in range(num_clusters):
            if clusters[c] != new_clusters[c]:
                condition = True
            clusters[c] = new_clusters[c]
                
        if condition is False:
            # No ha habido cambios de asignación
            # por lo que finaliza el bucle
            return clusters, clusters_centroid                                              

# Main    
clusters, centroids = kmeans(num_clusters, data, initial_method)
print(clusters)
print(centroids)

