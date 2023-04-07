__authors__ = '1496793,1606206,1498396'
__group__ = 'DJ.12'

import numpy as np
import Kmeans as km
import KNN as kn
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
from PIL import Image

def retrieval_by_color(images, Kmeans_results, color):
    images_color = []

    for i, kmean in enumerate(Kmeans_results):
        if color in kmean:
            images_color.append(images[i])

    array_images = np.array(images_color)
    visualize_retrieval(array_images, 10)

def retrieval_by_shape(images, Kmeans_results, shape):
    images_shape = []
    
    for i, kmean in enumerate(Kmeans_results):
        if shape == kmean:
            images_shape.append(images[i])
            
    array_images = np.array(images_shape)
    visualize_retrieval(array_images, 10)

def retrieval_combined(images, shape_labels, color_labels, shape, color): #podriamos dejarlo es facil
    imgs_with_shape_and_color = []
    i = 0
    for test_shape, test_color in zip(shape_labels, color_labels):
        if shape == test_shape and color in test_color:
            imgs_with_shape_and_color.append(images[i])
        i += 1
    images1 = np.array(imgs_with_shape_and_color)
    visualize_retrieval(images1, 10)
    
def get_shape_accuracy(knn_labels, gt):
    numbers = np.unique(knn_labels == gt, return_counts=True)[1][1]

    percentage_hits = (np.float64(numbers)/np.float64(len(knn_labels))) * np.float64(100)
    
    return percentage_hits

def get_color_accuracy(kmeans_labels, gt):
    hits = 0
    
    for pos, color in enumerate(kmeans_labels):
        count = 0
        color = np.unique(color)
        for col, ct in zip(color, gt[pos]):
            if col == ct:
                count += 1
        hits += count / len(gt[pos])

    percentage_hits = hits / len(kmeans_labels) * 100
    
    return percentage_hits

if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

## You can start coding your functions here

    kme = km.KMeans(test_imgs[30], 2)
    kme.fit()
    
    a = kn.KNN(train_imgs, train_class_labels)

    #prueba retrieval_by_color
    color = 'Blue'
    retrieval_by_color(test_imgs, test_color_labels, color)
    
    #prueba retrieval_by_shape
    shape = 'Jeans'
    retrieval_by_shape(test_imgs, test_class_labels, shape)
    
    #prueba retrieval_combined
    retrieval_combined(test_imgs,test_class_labels, test_color_labels, shape, color) 
    
    #prueba shape accuracy
    shape_acc = kn.KNN(train_imgs, train_class_labels)
    shape_acc.predict(test_imgs, 60)

    shape_percent = get_shape_accuracy(shape_acc.get_class(), test_class_labels)
    print("Percentatge: ", round(shape_percent,2), "%")
    
    #prueba color accuracy
    color_etiquetes = np.array([])
    kInicial = 2
    maxK = 30
    
    for k in range(kInicial, maxK):
        
        testColorAccuracy = km.KMeans(test_imgs[30], k)
        testColorAccuracy.fit()
        color_etiquetes = np.append(color_etiquetes, km.get_colors(testColorAccuracy.centroids))
        percentatge_color = get_color_accuracy(color_etiquetes, test_color_labels)
        
        print("k: ", k , "Percentatge color: ", round(percentatge_color,2))
        
    #prueba de las mejoras
    path = "./images/train/1529.jpg"
    img = Image.open(path)
    X = np.asarray(img)
    opciones = {}
    opciones['km_init'] = 'mid'
    #opciones['fitting'] = 'wcd'
    for j in range(10):
        for k in range (2,14):
            opciones['km_init'] = 'first'
            a = km.KMeans(X, K = k, options = opciones)
            a.fit()
            print("first: " + "k = " + str(k) + " n_iter = " + str(a.num_iter))
            
            opciones['km_init'] = 'last'
            a = km.KMeans(X, K = k, options = opciones)
            a.fit()
            print("last: " + "k = " + str(k) + " n_iter = " + str(a.num_iter))
            
            opciones['km_init'] = 'mid'
            a = km.KMeans(X, K = k, options = opciones)
            a.fit()
            print("half: " + "k = " + str(k) + " n_iter = " + str(a.num_iter))