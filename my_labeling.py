__authors__ = ['1636054', '1638922', '1636461']
__group__ = 'DJ.12'

import numpy as np
import Kmeans as km
import KNN as kn
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud
from PIL import Image
import matplotlib.pyplot as plt


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
            images[i] for i, (test_shape, test_color) in enumerate(zip(shape_labels, color_labels)) if
            shape == test_shape and color in test_color
        ]),
        topN
    )


def get_shape_accuracy(knn_labels, gt):
    unique_labels, label_counts = np.unique(knn_labels == gt, return_counts=True)
    return 100 * label_counts[1] / len(knn_labels)


def get_color_accuracy(kmeans_labels, gt):
    return sum(
        sum(col == ct for col, ct in zip(np.unique(color), gt[pos])) / len(gt[pos]) for pos, color in
        enumerate(kmeans_labels)
    ) / len(kmeans_labels) * 100


def print_plot(x, y, xtag, ytag, title, multiple=False):
    if multiple:
        for line in y:
            plt.plot(x, line)
    else:
        plt.plot(x, y)
    plt.xlabel(xtag)
    plt.ylabel(ytag)
    plt.title(title)
    plt.plot()
    plt.show()


def accuracy_test(min, max, shape=True, color=True):
    color_etiquetes = np.array([])
    x_axis = range(min, max)

    color_accuracy = []
    shape_accuracy = []

    for k in x_axis:
        # shape
        if shape:
            shape_acc = kn.KNN(train_imgs, train_class_labels, 3)
            shape_acc.predict(test_imgs, k)
            shape_percent = get_shape_accuracy(shape_acc.get_class(), test_class_labels)
            shape_accuracy.append(shape_percent)

        # color
        if color:
            color_accuracy_test = km.KMeans(test_imgs[30], k)
            color_accuracy_test.fit()
            color_etiquetes = np.append(color_etiquetes, km.get_colors(color_accuracy_test.centroids))
            percentatge_color = get_color_accuracy(color_etiquetes, test_color_labels)
            color_accuracy.append(round(percentatge_color, 2))

        print("k: ", k, "% color: ", round(percentatge_color, 2), "% shape: ", shape_percent)

    print_plot(x_axis, shape_accuracy, "k", "Accuracy %", "Shape Accuracy")
    print_plot(x_axis, color_accuracy, "k", "Accuracy %", "Color Accuracy")


def kmeans_init_test(path, min, max, passes):
    img = Image.open(path)
    x = np.asarray(img)
    opciones = {}

    x_axis = range(min, max)
    first_v_acc = []
    random_v_acc = []
    custom_v_acc = []
    for j in range(passes):
        random_v = []
        for k in x_axis:

            if j == 0:
                opciones['km_init'] = 'first'
                a = km.KMeans(x, K=k, options=opciones)
                a.fit()
                first_v_acc.append(a.num_iter)
                print("first: k =", k, "n_iter = ", a.num_iter)

                opciones['km_init'] = 'custom'
                c = km.KMeans(x, K=k, options=opciones)
                c.fit()
                custom_v_acc.append(c.num_iter)
                print("half: k =", k, "n_iter =", c.num_iter)

            opciones['km_init'] = 'random'
            b = km.KMeans(x, K=k, options=opciones)
            b.fit()
            random_v.append(b.num_iter)
            print("last: k =", k, "n_iter = ", b.num_iter)

        random_v_acc.append(random_v)

    print_plot(x_axis, first_v_acc, 'k', 'n iterations', "KMeans, km_init=first")
    print_plot(x_axis, random_v_acc, 'k', 'n iterations', "KMeans, km_init=random", True)
    print_plot(x_axis, custom_v_acc, 'k', 'n iterations', "KMeans, km_init=custom")


if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
        test_imgs, test_class_labels, test_color_labels = read_dataset()

    # List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    ## You can start coding your functions here

    kme = km.KMeans(train_imgs[0], 3)
    kme.fit()
    visualize_k_means(kme, [80, 60, 3])
    Plot3DCloud(kme)

    a = kn.KNN(train_imgs, train_class_labels, 3)

    # prueba retrieval_by_color
    color = 'Blue'
    retrieval_by_color(test_imgs, test_color_labels, color)

    # prueba retrieval_by_shape
    shape = 'Shirts'
    retrieval_by_shape(test_imgs, test_class_labels, shape)

    # prueba retrieval_combined
    retrieval_combined(test_imgs, test_class_labels, test_color_labels, shape, color)

    # km init
    kmeans_init_test('./images/train/1529.jpg', 2, 14, 10)

    # prueba accuracy
    accuracy_test(2, 18)
