_authors_ = ['1566595', '1567598', '1569198']
_group_ = 'DM.12'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist

class KNN:
    def _init_(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        train_data = np.array(train_data, dtype=np.float32)
        P, M, N, T = train_data.shape
        self.train_data = np.reshape(train_data, (P, M*N*T))


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        test_data = np.array(test_data, dtype=np.float32)
        P, M, N, T = test_data.shape
        self.test_data = np.reshape(test_data, (P, M*N*T))
        dist = np.argsort(cdist(self.test_data, self.train_data, 'euclidean'))[:,:k]
        self.neighbors = self.labels[dist]


    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        i = 0
        arr = np.array([])
        while i < len(self.neighbors):
            value, count = np.unique(self.neighbors[i], return_counts=True)
            arr = np.append(arr, value[np.argmax(count)])
            i += 1
        return arr

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()