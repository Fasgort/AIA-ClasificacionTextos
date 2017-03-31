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
        value_clase_unknown = 0
        #value_attrib_unknown = np.zeros([len(self.atributos)])
        
        # Poblando la matriz
        for v in range(len(entr)):
            value_clase = clas_entr[v]
            if value_clase == '?':
                value_clase_unknown += 1
                continue
            index_clase = np.where(self.clases==value_clase)
            self.train[0][index_clase] += 1
            for a in range(len(self.atributos)):
                value_attrib = entr[v][a]
                if value_attrib == '?':
                    #value_attrib_unknown[a] += 1
                    continue
                index_attrib = np.where(self.valores_atributos.get(self.atributos[a])==value_attrib)
                self.train[1][a][index_attrib][index_clase] += 1
                
        # Asignando probabilidades
        
        # Clases
        for c in range(len(self.clases)):
            if(self.train[0][c] != 0):
                self.train[0][c] = log(self.train[0][c]/(len(self.clases)-value_clase_unknown))
        
        # Atributos
        for a in range(len(self.atributos)):
            for v in range(len(self.valores_atributos.get(self.atributos[a]))):
                for c in range(len(self.clases)):
                    if self.train[0][c] != 0 and self.train[1][a][v][c] != 0:
                        self.train[1][a][v][c] = log((self.train[1][a][v][c] + self.k)/(self.train[0][c]) +
                                  self.k*len(self.valores_atributos.get(self.atributos[a])))
                        
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

        # *********** COMPLETA EL CรDIGO **************

# ---------------------------------------------------------------------------
        
from votes import *
clasificador = ClasificadorNaiveBayes(votos_atributo_clasificacion, votos_clases, votos_atributos, votos_valores_atributos, 1)
clasificador.entrena(votos_entr, votos_entr_clas, votos_valid, votos_valid_clas, False)