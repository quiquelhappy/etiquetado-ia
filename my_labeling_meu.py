__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
import Kmeans as km
import KNN as kn
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
#import cv2
import time
from Kmeans import get_colors

## You can start coding your functions here

def retrieval_by_color(images, Kmeans_results, color): #este seguro
    imgs_with_color = []

    for i, kmean in enumerate(Kmeans_results):
        if color in kmean:
            imgs_with_color.append(images[i])

    images1 = np.array(imgs_with_color)
    visualize_retrieval(images1, 10)

def retrieval_by_shape(images, Kmeans_results, shape): #este seguro
    imgs_with_shape = []
    for i, kmean in enumerate(Kmeans_results):
        if shape == kmean:
            imgs_with_shape.append(images[i])
    images1 = np.array(imgs_with_shape)
    visualize_retrieval(images1, 10)

def retrieval_combined(images, shape_labels, color_labels, shape, color): #podriamos dejarlo es facil
    imgs_with_shape_and_color = []
    i = 0
    for test_shape, test_color in zip(shape_labels, color_labels):
        if shape == test_shape and color in test_color:
            imgs_with_shape_and_color.append(images[i])
        i += 1
    images1 = np.array(imgs_with_shape_and_color)
    visualize_retrieval(images1, 10)

def get_shape_accuracy(knn_labels, gt): #este seguro
    trues = knn_labels == gt
    numbers = np.unique(trues, return_counts=True)[1][1]

    percentage = np.float64(numbers)/np.float64(len(knn_labels)) * np.float64(100)
    return percentage

def get_color_accuracy(kmeans_labels, gt): #este tamb podriamos dejarlo
    acc = 0
    for index, color in enumerate(kmeans_labels):
        count = 0
        color = np.unique(color)
        for c, ct in zip(color, gt[index]):
            if c == ct:
                count += 1
        acc += count / len(gt[index])

    percent = acc / len(kmeans_labels) * 100
    return percent

#==================plantearse si ponerlo==========================
def kmean_statistics(kmeans, Kmax, numero):     # train_images
    init_k = 2
    images = kmeans[:numero]
    for img in images:
        kmeans = km.KMeans(img, init_k)
        for k in range(init_k, Kmax):
            init_time = time.time()

            if k > init_k:
                kmeans.K = k

            init_time_fit = time.time()
            num_iterations = kmeans.fit()
            final_time_fit = time.time()

            time_fit = final_time_fit - init_time_fit
            print("Time fit k = {0} : {1} ".format(k, str(time_fit)))

            init_time_wcd = time.time()
            kmeans.whitinClassDistance()
            final_time_wcd = time.time()

            time_wcd = final_time_wcd - init_time_wcd
            print("Time to calculate WCD with k = {0} : {1}".format(k, str(time_wcd)))
            final_time_wcd = time.time()

            total_time = final_time_wcd - init_time
            print("Time in total for k = {0} : {1} in {2} iterations".format(k, str(total_time), num_iterations))
            print()
#===============================================================

if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    ########################### SET UP FUNCIONS ############################
    ########################################################################

    # INICIALITZACIÓ KMEANS

    # Opció 1
    kme = km.KMeans(test_imgs[30], 2)
    kme.fit()

    # Opció 2
    '''for img in test_imgs:
        kme = km.KMeans(img, 4)
        kme.fit()
        Kmeans_results = np.array(arr)'''

    # INICIALITZACIÓ KNN

    a = kn.KNN(train_imgs, train_class_labels)


    # RETRIEVALS

    color = 'Red'
    shape = 'Dresses'
    
    #test retruevals
    retrieval_by_color(test_imgs, test_color_labels, "Green")
    retrieval_by_color(test_imgs, test_color_labels, "Pink")
    retrieval_by_color(test_imgs, test_color_labels, "Red")

    retrieval_by_shape(test_imgs, test_class_labels, "Jeans")
    retrieval_by_shape(test_imgs, test_class_labels, "Sandal")
    retrieval_by_shape(test_imgs, test_class_labels, "Flip Flop")

    retrieval_combined(test_imgs,test_class_labels, test_color_labels, shape, color) #plantearse si este si o no


    # ACCURACY
    
    labels_colors = []

    color_etiquetes = np.array([])
    kInicial = 2
    maxK = 8
    
    for k in range(kInicial, maxK):
        testColorAccuracy = km.KMeans(test_imgs[30], k)
        testColorAccuracy.fit()
        color_etiquetes = np.append(color_etiquetes, km.get_colors(testColorAccuracy.centroids))
        percentatge_color = get_color_accuracy(color_etiquetes, test_color_labels)
        print("K, Color %: ", k, percentatge_color)

    a = kn.KNN(train_imgs, train_class_labels)
    a.predict(test_imgs, 5)         # imatges, K

    shape_percent = get_shape_accuracy(a.get_class(), test_class_labels)
    print("Shape Accuracy Percentatge: ", shape_percent)

    # STATISTICS ============ planterse si quitar este que es más rollo

    #initK = 2
    #kMax = 10
    #samples = 20
    #for img in test_imgs:
     #   kme = km.KMeans(img, initK)
     #   kme.fit()
     #    kmean_statistics(test_imgs, Kmax=10, numero=10)

    #kmean_statistics(test_imgs[430], kMax, samples)

    # KMEANS

    init_k = 20
    max_k = 100
    findMax = 10
    for i in range(init_k, max_k+1):
        opt = {'threshold': i, 'fitting': 'fisher'}
        kme = km.KMeans(test_imgs[400], K=5, options=opt)
        findK, fitIters = kme.find_bestK(max_K=findMax)

        print("Kmeans with threshold={0}, K is {1} and iterations are {2} ".format(i, findK, fitIters))

    opt = {'threshold': 20, 'fitting': 'wcd'}
    kme = km.KMeans(test_imgs[300], K=2, options=opt)
    findK, fitIters = kme.find_bestK(max_K=10)
    visualize_k_means(kme, [80,60,3])

    opt = {'threshold': 20, 'fitting': 'wcd'}
    kme = km.KMeans(test_imgs[300], K=2, options=opt)
    findK, fitIters = kme.find_bestK(max_K=10)
    visualize_k_means(kme, [80, 60, 3])

    opt = {'threshold': 20, 'fitting': 'wcd'}
    kme = km.KMeans(test_imgs[300], K=100, options=opt)
    findK, fitIters = kme.find_bestK(max_K=10)
    visualize_k_means(kme, [80,60,3])

    opt = {'threshold': 20, 'fitting': 'wcd'}
    kme = km.KMeans(test_imgs[300], K=800, options=opt)
    findK, fitIters = kme.find_bestK(max_K=10)
    visualize_k_means(kme, [80, 60, 3])

    ########################################################################
    ########################################################################