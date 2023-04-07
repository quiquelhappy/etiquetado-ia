__authors__ = ['1498396','1496793','1606206']
__group__ = 'DJ.12'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self,train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        dim = np.array([], dtype = float) #nos aseguramos de que sea float
        dim = np.shape(train_data) #cogemos las dimensiones 
        self.train_data = train_data.reshape(dim[0], 14400) #redimensionamos
        self.train_data = self.train_data.astype(type(np.inf))

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        dim = np.shape(test_data) #cogemos las dimensiones
        test_data = test_data.reshape(dim[0], dim[1]*dim[2]*dim[3]) #redimensionamos
        X = cdist(test_data, self.train_data, "euclidean")  #cogemos las distancias euclidianas
        Y = np.argsort(X, axis = 1) #ordenamos
        Z = Y[:,0:k] #cogemos los K valores
        self.neighbors = Z.astype(str) #pasamos el array a string
        for i in range(len(Z)):
            for j in range(len(Z[i])):
                self.neighbors[i][j] = self.labels[Z[i][j]]  #para cada id que tenemos ponemos su equivalente en prenda

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        masRepetidos = np.array([])
        
        for element in self.neighbors:
            prenda, repeticiones = np.unique(element, return_counts=True) #para cada elemento cogemos las veces que sale
            maximo = np.argmax(repeticiones) #cogemos el que mas sale
            masRepetidos = np.append(masRepetidos, prenda[maximo]) 
            
        return masRepetidos

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """
        
        self.get_k_neighbours(test_data, k)
        return self.get_class()