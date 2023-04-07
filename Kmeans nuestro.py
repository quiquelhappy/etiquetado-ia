__authors__ = ['1498396','1496793','1606206']
__group__ = 'DJ.12'

import numpy as np
import utils

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)
        self._init_centroids()

    def _init_X(self, X):
        dim = X.shape
        self.N = dim[0]*dim[1]
        self.X = np.resize(X,(dim[0]*dim[1],dim[2]))

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'wcd'
        if not 'limit' in options:
            options['limit'] = 20

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

    #Initialization of centroids
    def _init_centroids(self):
    
        self.old_centroids = np.array([])
        if self.options['km_init'].lower() == 'first':
            cont = 1
            i = 1
            self.old_centroids = np.append(self.old_centroids, self.X[0])
            self.old_centroids = np.resize(self.old_centroids, (cont, 3))
            dim = self.X.shape
            #busquem punts diferents dins la imatge X
            while(cont < self.K and cont < dim[0]):
                punt = np.resize(self.X[i],(1,3))
                j = 0
                #comprovem que no l'haguem trobat ja
                repetit = False
                while (j <= len(self.old_centroids) - 1 and not repetit):
                    if ((self.old_centroids[j] == punt).all()):
                        repetit = True
                    else:
                        j += 1
                if not repetit:
                    self.old_centroids = np.append(self.old_centroids, punt)
                    cont += 1
                    self.old_centroids = np.resize(self.old_centroids, (cont, 3))
                i += 1
            self.centroids = self.old_centroids.copy()
            
        elif self.options['km_init'].lower() == 'random':
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids =np.random.rand(self.K, self.X.shape[1])
            
        elif self.options['km_init'].lower() == 'last':
            indexes = np.sort(np.unique(self.X, return_index=True, axis=1)[1])
            self.centroids = self.old_centroids = self.X[indexes[0:self.K]]
            
        elif self.options['km_init'].lower() == 'half':
            indexes = np.sort(np.unique(self.X, return_index=True, axis=0)[1])
            lenInd = len(indexes)
            firstLen = round(lenInd / 2)
            randLen = round(lenInd / 2) + 1
            self.centroids = self.old_centroids = self.X[indexes[0:firstLen]]
            self.centroids = self.old_centroids = self.X[np.random.choice(indexes[randLen:], self.K, replace=False)]

    def get_labels(self):
        #calculem la distància de cada punt a cada centroid
        dist = distance(self.X, self.centroids)
        #guardem a labels l'index del vector fila de distancies als centroids
        self.labels = np.argmin(dist, axis = 1)

    def get_centroids(self):
        self.old_centroids = self.centroids.copy()
        for j in range(self.K):
            self.centroids[j] = np.mean(self.X[np.where(self.labels == j)],axis = 0, dtype = np.float64)

    #Checks if there is a difference between current and old centroids
    def converges(self):
        return (self.old_centroids == self.centroids).all()

    #Runs K-Means algorithm until it converges or until the number of iterations is smaller than the maximum number of iterations.
    def fit(self):
        self._init_centroids()
        while(self.num_iter < self.options['max_iter']):
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            if (self.converges()):
                break       

    def whitinClassDistance(self):
        
        if self.options['fitting'].lower() == 'wcd':
            self.WCD = self.intra_class_distance()

        elif self.options['fitting'].lower() == 'inter':
            self.WCD = self.inter_class_distance()

        elif self.options['fitting'].lower() == 'fisher':
            intra = self.intra_class_distance()
            inter = self.inter_class_distance()

            self.WCD = self.fisher_distance(intra, inter)

        return self.WCD
    
    def inter_class_distance(self):
        dist = distance(self.centroids, self.centroids)
        inter = 0
        for i, cent in enumerate(dist):
            inter += np.sum(cent[i:])
        inter /= len(self.centroids)

        return inter

    def intra_class_distance(self):
        #calculamos la distancia al centroide
        dist = distance(self.X,self.centroids)
        #nos quedamos el minimo
        dist = dist.min(axis = 1)
        #sumamos las distancias
        dist = np.sum(np.power(dist,2))
        #devolvemos dist/N
        return dist/len(self.X)

    def fisher_distance(self, intra, inter):
        fisher = intra/inter
        return fisher

    def find_bestK(self, max_K):

        #partimos con K = 2
        self.K = 2
        #hacemos primera iteración
        self.fit()
        self.WCD = self.whitinClassDistance()
        
        #recorremos hasta la max_K
        for self.K in range(3, max_K):
            #nos quedamos con el anterior para el calculo
            anterior = self.WCD
            self.fit()
            self.WCD = self.whitinClassDistance()
            
            quocient = 1 - (self.WCD / anterior)
            
            #si la bajada no ha llegado al 20% reducimos la k
            if (self.options['threshold'] / 100) > quocient:
                self.K -= 1
                break

        return self.K

def distance(X, C):
    #creamos el array con las dimensiones
    dist = np.zeros((X.shape[0],C.shape[0]))
    #calculamos la distancia
    for j in range(C.shape[0]):
        dist[:,j] = np.sqrt(np.sum(((X-C[j])**2), axis = 1))
    return dist

def get_colors(centroids):
    #cogemos las probabilidades de los colores
    CD = utils.get_color_prob(centroids)
    llista = []
    #nos vamos quedando con el que tenga más probabilidad
    for k in range(CD.shape[0]):
        llista.append(utils.colors[np.argmax(CD[k])])
    return llista