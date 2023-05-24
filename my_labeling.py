__authors__ = ['1636054', '1638922', '1636461']
__group__ = 'DJ.12'

import unittest

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
    match = 0
    for f in set(kmeans_labels[0]):
        match += int(f in gt)

    return match / len(gt) * 100


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


def accuracy_test(min, max, train_imgs, test_imgs, train_class_labels, test_class_labels, test_color_labels, shape=True,
                  color=True):
    color_etiquetes = np.array([])
    x_axis = range(min, max)

    color_accuracy = []
    shape_accuracy = []

    for k in x_axis:
        # shape
        if shape:
            shape_acc = kn.KNN(train_imgs, train_class_labels)
            shape_acc.predict(test_imgs, k)
            shape_percent = get_shape_accuracy(shape_acc.get_class(), test_class_labels)
            shape_accuracy.append(shape_percent)

            print("k: ", k, "% shape: ", shape_percent)

        # color
        if color:
            sample_size = 100
            accuracy_acc = 0
            for i in range(sample_size):
                color_accuracy_test = km.KMeans(test_imgs[i], k)
                color_accuracy_test.fit()
                color_etiquetes = km.get_colors(color_accuracy_test.centroids)
                accuracy_acc += get_color_accuracy(color_etiquetes, test_color_labels[i])
            percentage_color = accuracy_acc / sample_size
            color_accuracy.append(percentage_color)

            print("k: ", k, "% color: ", percentage_color)

    if shape:
        print_plot(x_axis, shape_accuracy, "k", "Accuracy %", "Shape Accuracy")

    if color:
        print_plot(x_axis, color_accuracy, "k", "Accuracy %", "Color Accuracy")


def find_best_k_test(min, max, fitting, test_imgs, test_color_labels, cluster=10):
    x_axis = range(min, max, 5)
    accuracy_color = []

    for threshold in x_axis:
        sample_size = 100
        accuracy_acc = 0
        for i in range(sample_size):
            color_accuracy_test = km.KMeans(test_imgs[i], 1, {
                'improvement_threshold': threshold,
                'fitting': fitting
            })
            color_accuracy_test.find_bestK(cluster)
            color_accuracy_test.fit()
            color_etiquetes = km.get_colors(color_accuracy_test.centroids)
            accuracy_acc += get_color_accuracy(color_etiquetes, test_color_labels[i])
        percentage_color = accuracy_acc / sample_size
        accuracy_color.append(percentage_color)
        print("threshold: ", threshold, "% color: ", percentage_color)

    print_plot(x_axis, accuracy_color, 'threshold', 'color accuracy', 'Find Best K ' + fitting)


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


class TestCases(unittest.TestCase):
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels = read_dataset()
    classes = list(set(list(train_class_labels) + list(test_class_labels)))
    color = 'Blue'
    shape = 'Shirts'

    def test_three_d_cloud(self):
        kme = km.KMeans(self.train_imgs[0], 3)
        kme.fit()
        visualize_k_means(kme, [80, 60, 3])
        Plot3DCloud(kme)

    def test_knn_init(self):
        a = kn.KNN(self.train_imgs, self.train_class_labels)

    def test_retrieval_by_color(self):
        retrieval_by_color(self.test_imgs, self.test_color_labels, self.color)

    def test_retrieval_by_shape(self):
        retrieval_by_shape(self.test_imgs, self.test_class_labels, self.shape)

    def test_retrieval_combined(self):
        retrieval_combined(self.test_imgs, self.test_class_labels, self.test_color_labels, self.shape, self.color)

    def test_find_best_k_wcd(self):
        find_best_k_test(0, 50, 'WCD', self.test_imgs, self.test_color_labels)

    def test_find_best_k_inter(self):
        find_best_k_test(0, 50, 'inter', self.test_imgs, self.test_color_labels)

    def test_find_best_k_fisher(self):
        find_best_k_test(0, 50, 'fisher', self.test_imgs, self.test_color_labels)

    def test_accuracy_color(self):
        accuracy_test(2, 17, self.train_imgs, self.test_imgs, self.train_class_labels, self.test_class_labels,
                      self.test_color_labels, False)

    def test_accuracy_shape(self):
        accuracy_test(2, 17, self.train_imgs, self.test_imgs, self.train_class_labels, self.test_class_labels,
                      self.test_color_labels, True, False)

    def test_kmeans_init(self):
        kmeans_init_test('./images/train/1529.jpg', 2, 14, 10)


if __name__ == '__main__':
    unittest.main()
