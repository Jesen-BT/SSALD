from skmultiflow.data.file_stream import FileStream
from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.utils.data_structures import ConfusionMatrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import copy as cp

import numpy as np
from scipy.stats import entropy
from collections import Counter
import random


class MCI_Kmeans:
    def __init__(self, k = 5, max_iter = 300, tol = 0.0001):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = []
        self.cluster_samples = {}
        self.Y = None

        self.labeled_center = []
        self.class_label = []
        self.weight = []

    def fit(self, X, Y):
        #Supervised clustering based on instances and their labels
        self.Y = Y
        self.centroids = self._init_centroids(X)
        prev_centroids = np.zeros(self.centroids.shape)
        labels = np.zeros(len(X))
        iters = 0

        while not self._has_converged(self.centroids, prev_centroids, self.tol, iters, self.max_iter):
            prev_centroids = np.copy(self.centroids)
            iters += 1

            # Assign labels to each data point based on centroids
            labels = self._assign_labels(X, self.centroids)
            self._get_cluster_samples(X, labels)

            # Update centroids based on the assigned labels
            self.centroids = self._update_centroids(X, labels, self.k, Y)

        labels = self._assign_labels(X, self.centroids)
        self._get_cluster_samples(X, labels)
        self.extract_summary()
        return labels

    def _init_centroids(self, X):
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]

    def _assign_labels(self, X, centroids):
        #This is the assigned cluster label, which cluster the sample belongs to. Not real class labels.
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels, k, Y):
        #Update the cluster centers according to the loss function.
        centroids = np.zeros((k, X.shape[1]))
        IPM = self.get_cluster_IMP()
        for i in range(k):
            IPM_i = IPM[i]
            weight = Y[labels == i]
            weight = [IPM_i if x != -1 else 1 for x in weight]
            centroids[i] = X[labels == i].mean(axis=0)
            centroids[i] = np.average(X[labels == i], axis=0, weights=weight)
        return centroids

    def _has_converged(self, centroids, prev_centroids, tol, iters, max_iter):
        if iters > max_iter:
            return True
        return np.linalg.norm(centroids - prev_centroids) < tol

    def _get_cluster_samples(self, X, labels):
        #Output which samples are in each cluster. At the same time, the label of the sample is also assigned to the cluster.
        self.cluster_samples = {}
        for i in range(self.k):
            self.cluster_samples[i] = []
        for i in range(len(X)):
            self.cluster_samples[labels[i]].append((X[i], self.Y[i]))

    def get_cluster_samples(self):
        return self.cluster_samples

    def get_cluster_IMP(self):
        #Calculate the Impurity Measures
        IMP_list = []
        for i in range(self.k):
            ADC = 0
            labels = [sample[1] for sample in self.cluster_samples[i]]
            n_labels = set(labels)
            result_dict = {element: sum([Counter(labels)[key] for key in Counter(labels).keys() if key != element and key != -1]) for element in n_labels}
            result_dict[-1] = 0
            for j in range(len(labels)):
                ADC = ADC + result_dict[labels[j]]
            labels = [x for x in labels if x >= 0]
            unique_labels, counts = np.unique(labels, return_counts=True)
            probs = counts / len(labels)
            IMP_list.append(entropy(probs)*ADC)
        return IMP_list

    def extract_summary(self):
        #Extract summaries for each cluster. The concept evolution was also considered in the original text, but since I focused on binary classification, this function was not implemented.
        for i in range(self.k):
            cluster_Y = [sample[1] for sample in self.cluster_samples[i]]
            counter = Counter(cluster_Y)
            del counter[-1]
            try:
                max_label = max(counter, key=counter.get)
                self.labeled_center.append(self.centroids[i])
                self.class_label.append(max_label)
                self.weight.append(counter[max_label]/sum(counter.values()))
            except ValueError:
                pass

    def get_summary(self):
        return self.labeled_center, self.class_label, self.weight

    def predict(self, X_test):
        N, d = X_test.shape
        pre = np.zeros(N)
        for i in range(N):
            distances = np.sqrt(np.sum((np.array(self.labeled_center) - X_test[i]) ** 2, axis=1))
            min_index = np.argmin(distances)
            pre[i] = self.class_label[min_index]
        return pre


class SmSCluster(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, label_budget=0.1, window_size=200, K=5, L=10, Q=5):
        self.window_size = window_size
        self.max_classifier = L
        self.K = K
        self.label_budget = label_budget
        self.Q = Q
        self.base_classifier = MCI_Kmeans(k=self.K)

        self.ensemble = []
        self.X_block = None
        self.Y_block = None
        self.i = -1

        self.centers = []
        self.center_labels = []
        self.center_weights = []

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        N, D = X.shape

        if self.i < 0:
            self.X_block = np.zeros((self.window_size, D))
            self.Y_block = np.zeros(self.window_size)
            self.i = 0

        for n in range(N):
            if random.random() < self.label_budget:
                self.X_block[self.i] = X[n]
                self.Y_block[self.i] = y[n]
            else:
                self.X_block[self.i] = X[n]
                self.Y_block[self.i] = -1
            self.i = self.i + 1

            if self.i == self.window_size:
                classifier = cp.deepcopy(self.base_classifier)
                classifier.fit(self.X_block, self.Y_block)
                self.ensemble.append(classifier)

                if len(self.ensemble) > self.max_classifier:
                    label_data = self.X_block[self.Y_block >= 0]
                    real_y = self.Y_block[self.Y_block >= 0]

                    index = 0
                    worst = 1

                    #Remove the worst one clustering model from the ensemble.
                    for i in range(len(self.ensemble)):
                        pre = self.ensemble[i].predict(label_data)
                        acc = accuracy_score(y_true=real_y, y_pred=pre)
                        if acc<worst:
                            worst = acc
                            index = i
                    self.ensemble.pop(index)
                self.i = 0
                self.extract_summary_from_ensemble()

    def extract_summary_from_ensemble(self):
        self.centers = []
        self.center_labels = []
        self.center_weights = []

        for i in range(len(self.ensemble)):
            center, center_label, center_weight = self.ensemble[i].get_summary()
            self.centers = self.centers + center
            self.center_labels = self.center_labels + center_label
            self.center_weights = self.center_weights + center_weight

    def Knn_vote(self, x):
        #Voting based on clustering model summary
        dis = np.linalg.norm(x - np.array(self.centers), axis=1)
        idx_sorted = np.argsort(dis)
        closest_idx = idx_sorted[:self.Q]
        closest_label = np.array(self.center_labels)[closest_idx]
        closest_weigt = np.array(self.center_weights)[closest_idx]

        label_weights_dict = {}

        for label, weight in zip(closest_label, closest_weigt):
            if label not in label_weights_dict:
                label_weights_dict[label] = 0
            label_weights_dict[label] += weight

        return max(label_weights_dict.items(), key=lambda x: x[1])[0]


    def predict_proba(self, X):
        N, D = X.shape
        votes = np.zeros(N)
        if len(self.ensemble) <= 0:
            return votes

        if len(self.centers) <= 0:
            return votes

        for n in range(N):
            votes[n] = self.Knn_vote(X[n])
        return votes


    def predict(self, X):
        votes = self.predict_proba(X)
        return (votes >= 0.5) * 1.


stream = FileStream("SINE.csv")
data, label = stream.next_sample(200)
tree = SmSCluster()
tree.partial_fit(data, label)
matrix = ConfusionMatrix(n_targets=2, dtype=np.float64)

count = 0
data_size = 0
result_list = []
t_list = []


while  stream.has_more_samples() and data_size < 100000:
    new_x, new_y = stream.next_sample()

    predict = tree.predict(new_x)
    if predict == new_y:
        count = count + 1

    matrix.update(new_y, int(predict))

    tree.partial_fit(new_x, new_y)

    data_size = data_size + 1

    if data_size % 100 == 0.:
        result_list.append(count/data_size)
        t_list.append(data_size)
        plt.plot(t_list, result_list, c='r', ls='-', marker='o', mec='b', mfc='w')
        plt.pause(0.1)

tree.extract_summary_from_ensemble()
plt.show()
