__authors__ = ['1636054','1638922','1636461']
__group__ = 'DJ.12'

import numpy as np
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        # Convierte los datos de entrenamiento en una matriz de tipo float y la asigna a D
        D = np.array(train_data, dtype=float)

        # Obtiene la forma de la matriz de entrenamiento y la asigna a D
        D = np.shape(train_data)

        # Reorganiza los datos de entrenamiento en una matriz de tamaño PxD
        # donde D = 4800*3, esto significa que cada imagen en train_data
        # se convierte en un punto con 4800x3 dimensiones
        self.train_data = train_data.reshape(D[0], -1)


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        D = np.shape(test_data)
        # Redimensionamos los datos de prueba para que sean una matriz de tamaño QxD
        test_data = test_data.reshape(test_data.shape[0], -1)
        # Calculamos las distancias euclidianas entre los datos de prueba y los datos de entrenamiento
        dist = cdist(test_data, self.train_data)
        # Obtenemos los índices de los k vecinos más cercanos para cada punto de prueba
        values = np.argsort(dist)[:, :k]
        # Obtenemos los valores de etiqueta correspondientes a los índices de los vecinos más cercanos
        self.neighbors = self.labels[values]

    def get_class(self):
        """
        Get the class by maximum voting.
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        # Create an empty array to store the most voted classes for each row
        list_fr = []

        for elementList in self.neighbors:
            list_arr = elementList.tolist()
            list_fr.append(max(list_arr, key=list_arr.count))

        return np.array(list_fr)


    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
