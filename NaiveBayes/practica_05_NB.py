# MII-AIA 2016-17
# Prรกctica del tema 6 - Parte 0 
# =============================

# Este trabajo estรก inspirado en el proyecto "Classification" de The Pacman
# Projects, desarrollados para el curso de Introducciรณn a la Inteligencia
# Artificial de la Universidad de Berkeley.

# Se trata de implementar el algoritmo Naive Bayes y de aplicarlo a dos
# problemas de aprendizaje para clasificaciรณn automรกtica. Estos problemas son:
# adivinar el partido polรญtico (republicano o demรณcrata) de un congresista USA
# a partir de lo votado a lo largo de un aรฑo, y reconocer un dรญgito a partir de
# una imagen del mismo escrito a mano.

# Conjuntos de datos
# ==================

# En este trabajo se manejarรกn dos conjuntos de datos, que serรกn usados para
# probar la implementaciรณn. A su vez cada conjunto de datos se distribuye en
# tres partes: conjunto de entrenamiento, conjunto de validaciรณn y conjunto de
# prueba. El primero de ellos se usarรก para el aprendizaje, el segundo para
# ajustar determinados parรกmetros de los clasificadores que finalmente se
# aprendan, y el tercero para medir el rendimiento de los mismos.

# Los datos que usaremos son:

#  - Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#    17 votaciones realizadas durante 1984. En votes.py estรกn estos datos, en
#    formato python. Este conjunto de datos estรก tomado de UCI Machine Learning
#    Repository, donde se puede encontrar mรกs informaciรณn sobre el mismo. Nรณtese
#    que en este conjunto de datos algunos valores figuran como desconocidos.

#  - Un conjunto de imรกgenes (en formato texto), con una gran cantidad de
#    dรญgitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#    base de datos MNIST. En digitdata.zip estรกn todos los datos en formato
#    comprimido. Cada imagen viene dada por 28x28 pรญxeles, y cada pixel vendrรก
#    representado por un caracter "espacio en blanco" (pixel blanco) o los
#    caracteres "+" (borde del dรญgito) o "#" (interior del dรญgito). En nuestro
#    caso trataremos ambos como un pixel negro (es decir, no distinguiremos
#    entre el borde y el interior). En cada conjunto, las imรกgenes vienen todas
#    seguidas en un fichero de texto, y las clasificaciones de cada imagen (es
#    decir, el nรบmero que representan) vienen en un fichero aparte, en el mismo
#    orden. Serรก necesario, por tanto, definir funciones python que lean esos
#    ficheros y obtengan los datos en el mismo formato python en el que se dan
#    los datos del punto anterior.

# Implementaciรณn del clasificador Naive Bayes
# ===========================================

# La implementaciรณn de ambos algoritmos deberรก realizarse completando el cรณdigo
# que se da mรกs abajo, siguiendo las indicaciones que aparecen en el mismo.

# Aunque el cรณdigo se aplicarรก a los conjuntos de datos anteriores, debe
# realizarse de manera independiente, para que sea posible aplicarlo a
# cualquier otro ejemplo de clasificaciรณn.

# Implementar el algoritmo Naive Bayes, tal y como se ha descrito en clase,
# usando suavizado de Laplace y logaritmos. La fase de ajuste en Naive Bayes
# consiste en encontrar el mejor k para el suavizado, de entre un conjunto
# de valores candidatos, probando los distintos rendimientos en el conjunto
# de validaciรณn (ver detalles en los comentarios del cรณdigo).

# El algoritmo debe poder tratar ejemplos con valores desconocidos en algรบn
# atributo (como los que aparecen en el caso de los votos). Para ello,
# simplemente ignorarlos (tanto para entrenamiento como para clasificaciรณn).

# Se pide dar el rendimiento (proporciรณn de aciertos) de cada clasificador
# sobre el conjunto de prueba proporcionado. Mostrar y comentar los resultados
# (incluyรฉndolos como comentarios al cรณdigo). En todos los casos, un
# rendimiento aceptable deberรญa estar por encima del 70% de aciertos sobre el
# conjunto de prueba.

# ----------------------------------------------------------------------------

# "*********** COMPLETA EL CรDIGO **************"

# ----------------------------------------------------------------------------
# Clase genรฉrica MetodoClasificacion
# ----------------------------------------------------------------------------

# EN ESTA PARTE NO SE PIDE NADA, PERO ES NECESARIO LEER LOS COMENTARIOS DEL
# CรDIGO. 

# Clase genรฉrica para los mรฉtodos de clasificaciรณn. Los mรฉtodos de
# clasificaciรณn que se piden deben ser subclases de esta clase genรฉrica. 

# NO MODIFICAR ESTA CLASE.

class MetodoClasificacion:
    """
    Clase base para mรฉtodos de clasificaciรณn
    """

    def __init__(self, atributo_clasificacion,clases,atributos,valores_atributos):

        """
        Argumentos de entrada al constructor (ver un caso concreto en votos.py)
         
        * atributo_clasificacion: es el atributo con los valores de clasificaciรณn. 
        * clases: lista de posibles valores del atributo de clasificaciรณn.  
        * atributos: lista de atributos, excepto el de clasificaciรณn. Tambiรฉn
                    denominados "caracterรญsticas". 
        * valores_atributos: diccionario que a cada atributo le asigna la
                             lista de sus posibles valores 
        """

        self.atributo_clasificacion=atributo_clasificacion
        self.clases = clases
        self.atributos=atributos
        self.valores_atributos=valores_atributos


    def entrena(self,entr,clas_entr,valid,clas_valid,autoajuste):
        """
        Mรฉtodo genรฉrico para entrenamiento y ajuste del clasificador. Deberรก
        ser definido para cada clasificador en particular. 
        
        Argumentos de entrada (ver un ejemplo en votos.py):

        * entr: ejemplos del conjunto de entrenamiento (sin incluir valor de
                clasificaciรณn) 
        * clas_entr: valores de clasificaciรณn de los ejemplos del conjunto de
                     entrenamiento
        * valid: ejemplos del conjujnto de validaciรณn (sin incluir valor de
                 clasificaciรณn)
        * clas_valid: valores de clasificaciรณn de los ejemplos del conjunto de 
                      validaciรณn
        * autoajuste: booleano para indicar si se hace autoajuste
        
        """
        abstract

    def clasifica(self, ejemplo):
        """
        Mรฉtodo genรฉrico para clasificaciรณn de un ejemplo, una vez entrenado el
        clasificador. Deberรก ser definido para cada clasificador en particular.

        Si se llama a este mรฉtodo sin haber entrenado previamente el
        clasificador, debe devolver un excepciรณn ClasificadorNoEntrenado
        (introducida mรกs abajo) 
        """
        abstract

# Excepciรณn que a de devolverse si se llama al mรฉtodo de clasificaciรณn antes de
# ser entrenado  
        
class ClasificadorNoEntrenado(Exception): pass
    
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Naive Bayes
# ----------------------------------------------------------------------------

# Implementar los mรฉtodos Naive Bayes de entrenamiento (con ajuste) y
# clasificaciรณn 

# LEER LOS COMENTARIOS AL CรDIGO

import numpy as np
from math import log
import copy

class ClasificadorNaiveBayes(MetodoClasificacion):

    def __init__(self,atributo_clasificacion,clases,atributos,valores_atributos,k=1):

        """ 
        Los argumentos de entrada al constructor son los mismos que los de la
        clase genรฉrica, junto con un parรกmetro k (cuyo valor por defecto es
        uno). Esta "k" es la que se tomarรก para el suavizado de Laplace,
        siempre que en el entrenamiento no se haga autoajuste (en ese caso, se
        tomarรก como "k" la que se decida en autoajuste).
        """
        
        self.atributo_clasificacion=atributo_clasificacion
        self.clases = clases
        self.atributos=atributos
        self.valores_atributos=valores_atributos
        self.k = k
        
        self.train = None
        
    def entrena(self,entr,clas_entr,valid,clas_valid,autoajuste=True):

        """ 
        Mรฉtodo para entrenamiento de Naive Bayes, que estima las probabilidades
        a partir del conjunto de entrenamiento y las almacena en forma
        logarรญtmica. A la estimaciรณn de las probabilidades se ha de aplicar
        suavizado de Laplace.  

        Si "autoajuste" es True (valor por defecto), el parรกmetro "k" del
        suavizado ha de elegirse de entre los siguientes valores candidatos,
        segรบn su rendimiento sobre el conjunto de validaciรณn:
        
        [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100] 

        Durante el ajuste, imprimir por pantalla los distintos rendimientos
        que se van obteniendo, y el "k" finalmente escogido 

        Si "autoajuste" es False, para el suavizado se tomarรก el "k" que se ha
        dado como argumento del constructor de la clase.

        Tener en cuenta que los ejemplos (tanto de entrenamiento como de
        clasificaciรณn) podrรญan tener algunos atributos con valores
        desconocidos. En ese caso, simplemente ignorar esos valores (pero no
        ignorar el ejemplo).
        """
        
        # Inicialización self.train
        self.train = [[],[]] # clases, atributos
        
        # Inicialización al vacío
        self.train[0] = np.zeros([len(self.clases)])
        self.train[1] = []
        for a in range(len(self.atributos)):
            self.train[1].append(np.zeros([len(self.valores_atributos.get(self.atributos[a])), len(self.clases)]))
        
        
        
        ### Poblando la matriz
        for v in range(len(entr)):
            value_clase = clas_entr[v]
            index_clase = self.clases.index(value_clase)
            self.train[0][index_clase] += 1
            for a in range(len(self.atributos)):
                value_attrib = entr[v][a]
                if value_attrib == '?':
                    continue
                index_attrib = self.valores_atributos.get(self.atributos[a]).index(value_attrib)
                self.train[1][a][index_attrib][index_clase] += 1
                         
                          
                          
        ### Cálculo del ajuste para Laplace
        if autoajuste == True:
            conj_ajustes = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100]
            rend_ajustes = []
            for ajuste in conj_ajustes:
                aux_train = copy.deepcopy(self.train)
                
                # Generación de probabilidades
                for a in range(len(self.atributos)):
                    for v in range(len(self.valores_atributos.get(self.atributos[a]))):
                        for c in range(len(self.clases)):
                            aux_train[1][a][v][c] = log((aux_train[1][a][v][c] + ajuste)/(aux_train[0][c] +
                                      ajuste*len(self.valores_atributos.get(self.atributos[a]))))
                            
                for c in range(len(self.clases)):
                    aux_train[0][c] = log(aux_train[0][c]/len(clas_entr))
            
                # Validación del ajuste
                pred_clas_valid = []
                for x in valid:
                    # Clasificación del conjunto de validación
                    prob_list_arr = np.zeros([len(self.clases)])
                    for c in range(len(self.clases)):
                        prob_list_arr[c] = aux_train[0][c]
                        for a in range(len(self.atributos)):
                            value_attrib = x[a]
                            if value_attrib != '?':
                                index_attrib = self.valores_atributos.get(self.atributos[a]).index(value_attrib)
                                prob_list_arr[c] += aux_train[1][a][index_attrib][c]
                
                    # Clasificación obtenida
                    pred_clas_valid.append(self.clases[np.argmax(prob_list_arr)])
            
                # Rendimiento obtenido con el ajuste
                aciertos = 0
                for c in range(len(clas_valid)):
                    if pred_clas_valid[c] == clas_valid[c]:
                        aciertos += 1
                accuracy = aciertos/len(clas_valid)
                rend_ajustes.append(accuracy)
            
            # Elección del mejor ajuste
            self.k = conj_ajustes[np.argmax(rend_ajustes)]
            print("Se ha estudiado el siguiente conjunto de ajustes:")
            print(conj_ajustes)
            print("La precisión de cada ajuste, respecto al conjunto de validación, son respectivamente, las siguientes:")
            print(rend_ajustes)
            print("Por lo tanto, se ha elegido el ajuste " + str(self.k) + ".")
            
            
                          
        ### Generación de probabilidades
        for a in range(len(self.atributos)):
            for v in range(len(self.valores_atributos.get(self.atributos[a]))):
                for c in range(len(self.clases)):
                    self.train[1][a][v][c] = log((self.train[1][a][v][c] + self.k)/(self.train[0][c] +
                              self.k*len(self.valores_atributos.get(self.atributos[a]))))
        
        for c in range(len(self.clases)):
            self.train[0][c] = log(self.train[0][c]/len(clas_entr))
                                                
    def clasifica(self,ejemplo):

        """ 
        Mรฉtodo para clasificaciรณn de ejemplos, usando el clasificador Naive
        Bayes obtenido previamente mediante el entrenamiento.

        Si se llama a este mรฉtodo sin haber entrenado previamente el
        clasificador, debe devolver una excepciรณn ClasificadorNoEntrenado

        Tener en cuenta que los ejemplos (tanto de entrenamiento como de
        clasificaciรณn) podrรญan tener algunos atributos con valores
        desconocidos. En ese caso, simplimente ignorar esos valores (pero no
        ignorar el ejemplo).
        """
        
        # Excepción si el clasificador no ha sido entrenado
        if self.train == None:
            raise ClasificadorNoEntrenado
        
        # Cálculo de probabilidades
        prob_list_arr = np.zeros([len(self.clases)])
        for c in range(len(self.clases)):
            prob_list_arr[c] = self.train[0][c]
            for a in range(len(self.atributos)):
                value_attrib = ejemplo[a]
                if value_attrib != '?':
                    index_attrib = self.valores_atributos.get(self.atributos[a]).index(value_attrib)
                    prob_list_arr[c] += self.train[1][a][index_attrib][c]

        return self.clases[np.argmax(prob_list_arr)]

# ---------------------------------------------------------------------------
        
### MAIN - Ejercicio 1 ###
from votes import *

# Se crea y entrena un clasificador para predecir el partido político de un congresista estadounidense
clasificadorVotos = ClasificadorNaiveBayes(votos_atributo_clasificacion,
                                           votos_clases, votos_atributos,
                                           votos_valores_atributos, 1)

print("Entrenando clasificador de votos.")
clasificadorVotos.entrena(votos_entr, votos_entr_clas, votos_valid,
                          votos_valid_clas, True)

# Clasificación
pred_test_clas = [clasificadorVotos.clasifica(x) for x in votos_test]

## Resultados clasificación (pred_test_clas)
#
#['democrata', 'republicano', 'democrata', 'republicano', 'democrata',
# 'republicano', 'democrata', 'democrata', 'republicano', 'republicano',
# 'democrata', 'republicano', 'democrata', 'democrata', 'democrata',
# 'republicano', 'republicano', 'republicano', 'democrata', 'democrata',
# 'democrata', 'republicano', 'democrata', 'democrata', 'republicano',
# 'republicano', 'republicano', 'republicano', 'democrata', 'republicano',
# 'republicano', 'republicano', 'democrata', 'democrata', 'republicano',
# 'democrata', 'republicano', 'republicano', 'democrata', 'democrata',
# 'republicano', 'democrata', 'republicano', 'democrata', 'republicano',
# 'democrata', 'democrata', 'democrata', 'democrata', 'republicano',
# 'democrata', 'republicano', 'republicano', 'republicano', 'democrata',
# 'republicano', 'republicano', 'republicano', 'democrata', 'republicano',
# 'democrata', 'republicano', 'republicano', 'democrata', 'republicano',
# 'republicano', 'democrata', 'democrata', 'republicano', 'democrata',
# 'democrata', 'democrata', 'republicano', 'democrata', 'democrata',
# 'democrata', 'democrata', 'democrata', 'democrata', 'republicano',
# 'democrata', 'democrata', 'republicano', 'democrata', 'republicano',
# 'republicano', 'republicano']
#
###

# Medidas de rendimiento
aciertos = 0
for c in range(len(votos_test_clas)):
    if pred_test_clas[c] == votos_test_clas[c]:
        aciertos += 1
accuracy = aciertos/len(votos_test_clas)
print("La precisión del clasificador ha sido del " + str(accuracy*100) + " % para la predicción del partido político.")
print()
print()

## Resultados ajuste + precisión (accuracy)
#
# Entrenando clasificador de votos.
# Se ha estudiado el siguiente conjunto de ajustes:
# [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100]
# La precisión de cada ajuste, respecto al conjunto de validación, son respectivamente, las siguientes:
# [0.9710144927536232, 0.9710144927536232, 0.9710144927536232, 0.9710144927536232, 0.9710144927536232, 0.9710144927536232, 0.9710144927536232, 0.9710144927536232, 0.9710144927536232, 0.9420289855072463, 0.9420289855072463]
# Por lo tanto, se ha elegido el ajuste 0.001.
# La precisión del clasificador ha sido del 83.9080459770115 % para la predicción del partido político.
#
###



### MAIN - Ejercicio 2 ###
from digitdata import readDigitData
(digitos_atributo_clasificacion, digitos_clases, digitos_atributos,
 digitos_valores_atributos, digitos_entr, digitos_entr_clas, digitos_valid,
 digitos_valid_clas, digitos_test, digitos_test_clas) = readDigitData()

# Se crea y entrena un clasificador para predecir un dígito escrito
clasificadorDigitos = ClasificadorNaiveBayes(digitos_atributo_clasificacion,
                                             digitos_clases, digitos_atributos,
                                             digitos_valores_atributos, 1)

print("Entrenando clasificador de dígitos.")
clasificadorDigitos.entrena(digitos_entr, digitos_entr_clas, digitos_valid,
                            digitos_valid_clas, True)

# Clasificación
pred_test_clas = [clasificadorDigitos.clasifica(x) for x in digitos_test]

## Resultados clasificación (pred_test_clas)
#
#['7', '0', '2', '3', '1', '9', '7', '8', '1', '0', '4', '1', '9', '9', '4',
# '9', '2', '2', '8', '1', '3', '7', '9', '4', '8', '1', '8', '1', '3', '8',
# '1', '2', '8', '8', '0', '6', '2', '1', '1', '9', '1', '5', '3', '4', '8',
# '9', '5', '0', '9', '2', '2', '4', '9', '2', '1', '9', '2', '4', '9', '4',
# '4', '0', '9', '9', '2', '2', '3', '3', '5', '3', '5', '7', '7', '5', '8',
# '1', '2', '9', '6', '6', '4', '9', '5', '1', '0', '6', '9', '5', '9', '5',
# '9', '9', '1', '8', '0', '3', '9', '1', '3', '6', '7', '2', '5', '9', '7',
# '9', '6', '3', '6', '3', '7', '4', '6', '5', '8', '9', '9', '8', '8', '8',
# '8', '6', '7', '6', '8', '9', '7', '9', '7', '1', '9', '5', '2', '7', '3',
# '5', '1', '1', '2', '1', '4', '7', '4', '7', '3', '4', '5', '4', '0', '8',
# '3', '6', '9', '5', '0', '2', '7', '9', '4', '8', '4', '6', '6', '4', '7',
# '9', '3', '4', '5', '9', '8', '7', '3', '9', '2', '7', '5', '2', '4', '1',
# '1', '1', '6', '9', '2', '9', '7', '2', '0', '1', '5', '4', '4', '1', '9',
# '0', '6', '6', '4', '6', '5', '1', '5', '0', '9', '3', '8', '2', '9', '5',
# '2', '1', '8', '1', '1', '3', '7', '9', '0', '3', '0', '9', '4', '0', '6',
# '8', '2', '2', '3', '8', '4', '0', '9', '6', '5', '3', '1', '2', '1', '3',
# '1', '7', '9', '5', '7', '2', '0', '0', '3', '8', '1', '2', '3', '4', '1',
# '9', '3', '1', '5', '8', '1', '0', '2', '4', '4', '3', '6', '9', '8', '2',
# '9', '0', '4', '8', '4', '9', '7', '9', '3', '4', '1', '5', '4', '2', '3',
# '5', '8', '8', '8', '9', '3', '3', '6', '6', '0', '1', '6', '0', '3', '7',
# '4', '4', '1', '2', '9', '1', '4', '6', '4', '9', '7', '9', '8', '4', '2',
# '5', '1', '9', '1', '3', '1', '7', '9', '4', '8', '8', '2', '9', '9', '1',
# '9', '3', '6', '0', '5', '2', '2', '6', '1', '3', '5', '2', '4', '9', '1',
# '6', '7', '9', '2', '2', '1', '1', '2', '8', '3', '7', '8', '4', '1', '7',
# '1', '7', '6', '7', '2', '2', '7', '3', '1', '7', '5', '8', '2', '6', '2',
# '8', '5', '6', '5', '0', '9', '2', '4', '6', '9', '9', '7', '6', '6', '8',
# '0', '4', '1', '3', '3', '2', '9', '1', '8', '0', '6', '7', '7', '1', '8',
# '5', '5', '2', '0', '1', '6', '5', '1', '4', '9', '8', '0', '9', '9', '4',
# '6', '5', '4', '9', '1', '8', '3', '4', '9', '9', '1', '3', '2', '3', '1',
# '9', '5', '4', '0', '1', '9', '8', '3', '8', '2', '0', '2', '5', '1', '9',
# '2', '2', '9', '9', '5', '9', '6', '0', '6', '2', '5', '4', '2', '7', '3',
# '5', '3', '9', '0', '6', '8', '5', '3', '5', '8', '6', '3', '7', '1', '3',
# '3', '9', '6', '1', '1', '2', '9', '0', '4', '3', '3', '6', '9', '5', '9',
# '3', '7', '7', '7', '3', '1', '9', '8', '3', '0', '7', '2', '7', '9', '4',
# '5', '4', '9', '3', '8', '1', '4', '0', '2', '3', '7', '5', '9', '8', '8',
# '0', '0', '6', '1', '4', '7', '3', '9', '0', '6', '0', '2', '6', '2', '3',
# '7', '8', '4', '7', '7', '4', '2', '9', '1', '6', '5', '2', '4', '8', '9',
# '1', '9', '4', '0', '3', '8', '4', '3', '7', '7', '0', '7', '8', '8', '4',
# '0', '9', '8', '8', '2', '4', '7', '6', '6', '5', '4', '9', '1', '8', '8',
# '2', '3', '6', '3', '0', '0', '3', '7', '6', '9', '7', '9', '9', '5', '4',
# '3', '7', '6', '1', '2', '3', '7', '5', '3', '6', '0', '3', '3', '8', '9',
# '3', '0', '3', '5', '0', '2', '0', '9', '0', '7', '4', '5', '4', '3', '5',
# '1', '9', '6', '1', '7', '5', '4', '5', '8', '5', '4', '5', '2', '1', '1',
# '9', '1', '9', '9', '4', '0', '8', '4', '5', '2', '4', '2', '1', '2', '1',
# '1', '3', '6', '9', '3', '4', '9', '1', '9', '8', '5', '7', '5', '1', '1',
# '8', '6', '5', '3', '4', '4', '7', '2', '3', '1', '6', '5', '8', '6', '2',
# '3', '5', '0', '5', '3', '7', '6', '9', '6', '7', '0', '4', '8', '7', '1',
# '7', '4', '1', '0', '5', '7', '2', '0', '0', '9', '1', '1', '0', '4', '8',
# '4', '9', '4', '0', '4', '6', '0', '7', '1', '1', '3', '3', '9', '2', '7',
# '4', '1', '2', '3', '0', '8', '1', '3', '9', '6', '9', '3', '5', '0', '2',
# '7', '4', '5', '1', '2', '5', '8', '5', '8', '5', '1', '5', '0', '3', '0',
# '3', '1', '4', '0', '3', '7', '2', '4', '1', '5', '0', '7', '0', '4', '3',
# '1', '9', '4', '7', '7', '5', '9', '9', '9', '3', '4', '1', '7', '9', '0',
# '2', '0', '3', '3', '7', '4', '4', '2', '3', '3', '7', '7', '0', '0', '7',
# '5', '2', '9', '5', '7', '4', '9', '2', '6', '6', '1', '9', '6', '9', '2',
# '9', '0', '8', '3', '1', '1', '6', '3', '5', '1', '1', '1', '3', '1', '3',
# '3', '0', '8', '2', '1', '3', '5', '0', '7', '4', '9', '9', '6', '9', '6',
# '9', '3', '6', '6', '8', '5', '1', '4', '2', '4', '9', '5', '1', '1', '9',
# '0', '8', '4', '9', '4', '7', '1', '8', '3', '5', '6', '9', '4', '9', '1',
# '1', '6', '7', '6', '3', '2', '2', '0', '3', '9', '2', '3', '1', '0', '3',
# '2', '4', '5', '4', '9', '6', '9', '0', '6', '1', '5', '5', '8', '3', '8',
# '2', '6', '8', '0', '7', '4', '6', '1', '3', '4', '7', '5', '2', '3', '9',
# '2', '3', '2', '7', '1', '7', '2', '6', '6', '1', '5', '7', '8', '6', '0',
# '1', '8', '2', '4', '7', '7', '6', '5', '3', '5', '2', '4', '2', '4', '5',
# '8', '8', '3', '4', '9', '2', '7', '5', '9', '6', '3', '6', '0', '3', '6',
# '7', '3', '6', '4', '9', '4', '6', '5', '3', '0', '4', '1', '0', '1', '9',
# '6', '2', '9', '1', '1', '0', '6', '3', '9', '5']
#
###

# Medidas de rendimiento
aciertos = 0
for c in range(len(digitos_test_clas)):
    if pred_test_clas[c] == digitos_test_clas[c]:
        aciertos += 1
accuracy = aciertos/len(digitos_test_clas)
print("La precisión del clasificador ha sido del " + str(accuracy*100) + " % para la predicción de dígitos.")
print()
print()

## Resultados ajuste + precisión (accuracy)
#
# Entrenando clasificador de dígitos.
# Se ha estudiado el siguiente conjunto de ajustes:
# [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100]
# La precisión de cada ajuste, respecto al conjunto de validación, son respectivamente, las siguientes:
# [0.822, 0.819, 0.817, 0.817, 0.817, 0.818, 0.81, 0.799, 0.791, 0.764, 0.716]
# Por lo tanto, se ha elegido el ajuste 0.001.
# La precisión del clasificador ha sido del 77.4 % para la predicción de dígitos.
#
###
