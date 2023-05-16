__authors__ = ['1636461', '1636054', '1638922']
__group__ = 'DJ.12'

import numpy as np
import utils


# frequent problems/doubts we encountered
# - https://stackoverflow.com/questions/10580676/comparing-two-numpy-arrays-for-equality-element-wise
# - https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
# - https://numpy.org/doc/stable/reference/generated/numpy.shape.html
# - https://note.nkmk.me/en/python-numpy-ndarray-ndim-shape-size/
# - https://stackoverflow.com/questions/40312013/check-type-within-numpy-array

class KMeans:
    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.old_centroids = None
        self.labels = None
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    def _init_X(self, X: np.matrix):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        # convert all values instead of allowing implicit casting on the rest of the methods,
        # this speeds up the execution a little bit
        X = X.astype('float64')

        # compute N and D
        self.N = X.shape[1] * X.shape[0]
        self.D = 3

        # consider dim>=3
        if len(X.shape) >= 3:
            # TODO check if len(X.shape)-1 is a suitable option to resize to the length of the last dimension
            self.X = np.reshape(X, (self.N, X.shape[len(X.shape) - 1]))
        else:
            self.X = np.reshape(X, (self.N, self.D))

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        # improvement threshold for finding best_k
        if options is None:
            options = {}
        # If your methods need any other parameter you can add it to the options dictionary
        # improvement threshold for k-means
        if 'improvement_threshold' not in options:
            options['improvement_threshold'] = 20

        # original settings dict
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        self.options = options

    # Initialization of centroids
    def _init_centroids(self):
        """
        Initialization of centroids
        """
        self.old_centroids = np.array([])
        if self.options['km_init'].lower() == 'first':
            i = 1
            unique_count = 1

            self.old_centroids = np.append(self.old_centroids, self.X[0])
            self.old_centroids = np.resize(self.old_centroids, (unique_count, 3))
            # iterate over the first k elements of the matrix, without
            # exceeding the matrix length of that dimension
            while unique_count < self.K and unique_count < self.X.shape[0]:
                # get RGB values of the pixel
                point = np.resize(self.X[i], (1, 3))
                unique = True
                for old_point in self.old_centroids:
                    # we use array equiv in order to evaluate
                    # true in the following cases:
                    # [255. 255. 255.] [[255 255 255]]
                    # note the first array can be structured
                    # to match the other array, that's exactly
                    # what array_equiv does
                    if np.array_equiv(old_point, point):
                        unique = False
                        break

                if unique:
                    unique_count += 1
                    self.old_centroids = np.append(self.old_centroids, point)
                    self.old_centroids = np.resize(self.old_centroids, (unique_count, 3))
                i += 1

            self.centroids = self.old_centroids.copy()
        elif self.options['km_init'].lower() == 'random':
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            # does not affect runtime, and speeds up execution
            self.old_centroids = self.centroids.copy()

        elif self.options['km_init'].lower() == 'custom':
            idx = np.sort(np.unique(self.X, return_index=True, axis=0)[1])
            self.old_centroids = self.X[idx[(len(idx) - self.K):]]
            self.centroids = self.old_centroids.copy()

    # ----- KMEANS METHODS -----

    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        # get distance from all points to centroid, and returns the closest
        self.labels = np.argmin(distance(self.X, self.centroids), axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids.copy()  # we copy to avoid mem reference
        for i in range(self.K):
            # gets the mean of all values with label j (i), which satisfies the formula
            # in the PDF of point 2, section KMeans
            self.centroids[i] = np.mean(self.X[np.where(i == self.labels)], dtype=np.float64, axis=0)

    # Checks if there is a difference between current and old centroids
    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.array_equiv(self.old_centroids, self.centroids)

    # ----- REST OF KMEANS METHODS -----

    # Runs K-Means algorithm until it converges or until the number of iterations is smaller than the maximum number
    # of iterations.
    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        self._init_centroids()
        while self.num_iter < self.options['max_iter']:
            self.num_iter += 1
            self.get_labels()
            self.get_centroids()
            if self.converges():
                break

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        if self.options['fitting'].lower() == 'wcd':
            # get nearest, and compute average
            return np.average(np.power(distance(self.X, self.centroids).min(axis=1), 2))

        elif self.options['fitting'].lower() == 'fisher':
            intra = self.intra_class_distance()
            inter = self.inter_class_distance()
            self.WCD = self.fisher_distance(intra, inter)

        elif self.options['fitting'].lower() == 'inter':
            self.WCD = self.inter_class_distance()


        return self.WCD

    def find_bestK(self, max_K, threshold=20):
        """
        Sets the best K by analyzing the results up to 'max_K' clusters.
        The threshold parameter determines the minimum improvement required for K.

        Args:
            max_K (int): The maximum number of clusters to analyze.
            threshold (float): The minimum improvement threshold as a percentage (default: 20%).

        Returns:
            int: The best value of K.
        """
        self.K = 2
        wcd = 0.0

        for self.K in range(2, max_K):
            prev = wcd
            self.fit()
            wcd = self.withinClassDistance()

            if self.K > 2:
                improvement = 1 - (wcd / prev)
                if improvement < (threshold / 100.0):
                    self.K -= 1
                    break

        return self.K


# from KMeans
def distance(X: np.array, C: np.array) -> np.array:
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set and the j-th point of the second set
    """
    # In the expression (X[:, None, :] - C)**2, the resulting array has shape (N, K, D), where N is the number
    # of data points, K is the number of centroids, and D is the dimensionality of the data.
    # The third axis (axis 2) represents the dimensionality of the data.
    return np.sqrt(((X[:, None, :] - C)**2).sum(axis=2))


# ----

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    color_dist = utils.get_color_prob(centroids)
    return [utils.colors[np.argmax(color_dist[k])] for k in range(color_dist.shape[0])]
