__authors__ = ['1636054', '1638922', '1636461']
__group__ = 'DJ.12'

import unittest

import numpy as np
import Kmeans as km
import KNN as kn
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud
from PIL import Image
import matplotlib.pyplot as plt


def get_shape_accuracy(knn_labels, gt):
    unique_labels, label_counts = np.unique(knn_labels == gt, return_counts=True)
    return label_counts[1] / len(knn_labels) * 100


def get_color_accuracy(kmeans_labels, gt):
    match = 0
    for f in set(kmeans_labels):
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
        print("find best k", "threshold: ", threshold, "% color: ", percentage_color)

    print_plot(x_axis, accuracy_color, 'threshold', 'color accuracy', 'Find Best K ' + fitting)


def kmeans_init_test(path, min, max, passes, method):
    img = Image.open(path)
    x = np.asarray(img)
    opciones = {}

    x_axis = range(min, max)
    acc = []
    for j in range(passes):
        pass_acc = []
        for k in x_axis:
            opciones['km_init'] = method
            a = km.KMeans(x, K=k, options=opciones)
            a.fit()
            pass_acc.append(a.num_iter)
            print("kmeans init", method, "k", k, "n_iter", a.num_iter)

        acc.append(pass_acc)

    print_plot(x_axis, acc, 'k', 'n iterations', "KMeans, km_init="+method, True)


class TestCases(unittest.TestCase):
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels = read_dataset()
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # init
    init_source = './images/train/1529.jpg'

    # retrieval
    color = 'Blue'
    shape = 'Shirts'
    top_n = 10

    # accuracy
    accuracy_sample_size = 100
    accuracy_min_k = 2
    accuracy_max_k = 17

    # quick KNN init
    def test_a_knn_init(self):
        kn.KNN(self.train_imgs, self.train_class_labels)

    # visualize a 3D dot cloud
    def test_a_three_d_cloud(self):
        kme = km.KMeans(self.train_imgs[0], 3)
        kme.fit()
        visualize_k_means(kme, [80, 60, 3])
        Plot3DCloud(kme)

    # return top_n images matching the color feature
    def test_b_retrieval_by_color(self):
        visualize_retrieval(
            np.array([
                self.test_imgs[i] for i, kmean in enumerate(self.test_color_labels) if self.color in kmean
            ]),
            self.top_n
        )

    # return top_n images matching the shape feature
    def test_b_retrieval_by_shape(self):
        visualize_retrieval(
            np.array([
                self.test_imgs[i] for i, kmean in enumerate(self.test_class_labels) if self.shape == kmean
            ]),
            self.top_n
        )

    # return top_n images matching both the color and shape features specified
    def test_b_retrieval_combined(self):
        visualize_retrieval(
            np.array([
                self.test_imgs[i] for i, (test_shape, test_color) in enumerate(zip(self.test_class_labels, self.test_color_labels)) if
                self.shape == test_shape and self.color in test_color
            ]),
            self.top_n
        )

    # with this test, we should see that the first init usually causes a
    # high n of iterations, so it is suboptimal. As with any init technique,
    # when we increase k, the n of iterations will usually increase proportionally,
    # since we need to initialize more centroids
    def test_c_kmeans_init_first(self):
        kmeans_init_test(self.init_source, 2, 14, 10, 'first')

    # with this test, we should see that the custom init usually causes the
    # least n of iterations, so it is usually most optimal. As with any init technique,
    # when we increase k, the n of iterations will usually increase proportionally,
    # since we need to initialize more centroids
    def test_c_kmeans_init_custom(self):
        kmeans_init_test(self.init_source, 2, 14, 10, 'custom')

    # with this test, we should see that the random init usually causes the
    # most n of iterations, so it is suboptimal. As with any init technique,
    # when we increase k, the n of iterations will usually increase proportionally,
    # since we need to initialize more centroids
    def test_c_kmeans_init_random(self):
        kmeans_init_test(self.init_source, 2, 14, 10, 'random')

    def test_d_find_best_k_wcd(self):
        find_best_k_test(0, 50, 'WCD', self.test_imgs, self.test_color_labels)

    def test_d_find_best_k_inter(self):
        find_best_k_test(0, 50, 'inter', self.test_imgs, self.test_color_labels)

    def test_d_find_best_k_fisher(self):
        find_best_k_test(0, 50, 'fisher', self.test_imgs, self.test_color_labels)

    # this test compares the KMeans result to the ground truth, with
    # multiple values of k, so we can see performance based on the k value.
    # since we are able to run a subset of the sample, we get the first
    # n accuracy_sample_size images to test our accuracy
    def test_e_accuracy_color(self):
        x_axis = range(self.accuracy_min_k, self.accuracy_max_k)
        color_accuracy = []
        for k in x_axis:
            accuracy_acc = 0
            for i in range(self.accuracy_sample_size):
                color_accuracy_test = km.KMeans(self.test_imgs[i], k)
                color_accuracy_test.fit()
                color_etiquetes = km.get_colors(color_accuracy_test.centroids)
                accuracy_acc += get_color_accuracy(color_etiquetes, self.test_color_labels[i])
            percentage_color = accuracy_acc / self.accuracy_sample_size

            color_accuracy.append(percentage_color)
            print("color accuracy", "k", k, "% acc", percentage_color)

        print_plot(x_axis, color_accuracy, "k", "Accuracy %", "Color Accuracy")

    # this test compares the KNN result to the expected ground truth, with
    # multiple values of k, so we can see accuracy performance by k value
    def test_e_accuracy_shape(self):
        x_axis = range(self.accuracy_min_k, self.accuracy_max_k)
        shape_accuracy = []

        for k in x_axis:
            shape_acc = kn.KNN(self.train_imgs, self.train_class_labels)
            shape_acc.predict(self.test_imgs, k)
            shape_percent = get_shape_accuracy(shape_acc.get_class(), self.test_class_labels)

            shape_accuracy.append(shape_percent)
            print("shape accuracy", "k", k, "% acc", shape_percent)

        print_plot(x_axis, shape_accuracy, "k", "Accuracy %", "Shape Accuracy")


if __name__ == '__main__':
    unittest.main()
