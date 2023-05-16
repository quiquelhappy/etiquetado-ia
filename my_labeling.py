__authors__ = '1496793,1606206,1498396'
__group__ = 'DJ.12'

import numpy as np
import Kmeans as km
import KNN as kn
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
from PIL import Image


def retrieval_by_color(images, Kmeans_results, color, topN=10):
    visualize_retrieval(
        np.array([
            images[i] for i, kmean in enumerate(Kmeans_results) if color in kmean
        ]),
        topN
    )


def retrieval_by_shape(images, Kmeans_results, shape, topN=10):
    visualize_retrieval(
        np.array([
            images[i] for i, kmean in enumerate(Kmeans_results) if shape == kmean
        ]),
        topN
    )


def retrieval_combined(images, shape_labels, color_labels, shape, color, topN=10):  # podriamos dejarlo es facil
    visualize_retrieval(
        np.array([
            images[i] for i, (test_shape, test_color) in enumerate(zip(shape_labels, color_labels)) if shape == test_shape and color in test_color
        ]),
        topN
    )


def get_shape_accuracy(knn_labels, gt):
    return (np.float64(np.unique(knn_labels == gt, return_counts=True)[1][1]) / np.float64(
        len(knn_labels)
    )) * np.float64(100)


def get_color_accuracy(kmeans_labels, gt):
    return sum(
        sum(col == ct for col, ct in zip(np.unique(color), gt[pos])) / len(gt[pos]) for pos, color in enumerate(kmeans_labels)
      ) / len(kmeans_labels) * 100


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
        test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/',
                                                                       gt_json='./images/gt.json')

    # List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    ## You can start coding your functions here

    kme = km.KMeans(test_imgs[30], 2)
    kme.fit()

    a = kn.KNN(train_imgs, train_class_labels)

    # prueba retrieval_by_color
    color = 'Blue'
    retrieval_by_color(test_imgs, test_color_labels, color)

    # prueba retrieval_by_shape
    shape = 'Jeans'
    retrieval_by_shape(test_imgs, test_class_labels, shape)

    # prueba retrieval_combined
    retrieval_combined(test_imgs, test_class_labels, test_color_labels, shape, color)

    # prueba shape accuracy
    shape_acc = kn.KNN(train_imgs, train_class_labels)
    shape_acc.predict(test_imgs, 60)

    shape_percent = get_shape_accuracy(shape_acc.get_class(), test_class_labels)
    print("Percentatge: ", round(shape_percent, 2), "%")

    # prueba color accuracy
    color_etiquetes = np.array([])
    kInicial = 2
    maxK = 30

    for k in range(kInicial, maxK):
        testColorAccuracy = km.KMeans(test_imgs[30], k)
        testColorAccuracy.fit()
        color_etiquetes = np.append(color_etiquetes, km.get_colors(testColorAccuracy.centroids))
        percentatge_color = get_color_accuracy(color_etiquetes, test_color_labels)

        print("k: ", k, "Percentatge color: ", round(percentatge_color, 2))

    # prueba de las mejoras
    path = "./images/train/1529.jpg"
    img = Image.open(path)
    X = np.asarray(img)
    opciones = {}
    opciones['km_init'] = 'mid'
    # opciones['fitting'] = 'wcd'
    for j in range(10):
        for k in range(2, 14):
            opciones['km_init'] = 'first'
            a = km.KMeans(X, K=k, options=opciones)
            a.fit()
            print("first: " + "k = " + str(k) + " n_iter = " + str(a.num_iter))

            opciones['km_init'] = 'last'
            a = km.KMeans(X, K=k, options=opciones)
            a.fit()
            print("last: " + "k = " + str(k) + " n_iter = " + str(a.num_iter))

            opciones['km_init'] = 'mid'
            a = km.KMeans(X, K=k, options=opciones)
            a.fit()
            print("half: " + "k = " + str(k) + " n_iter = " + str(a.num_iter))
