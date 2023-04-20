import copy

from skmultiflow.data.file_stream import FileStream
from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.utils.data_structures import ConfusionMatrix
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

import random
import copy as cp

from sklearn.cluster import KMeans

class KMeansCluster:
    #This is the base classifier for the unsupervised model, only clustering.
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.labels = None
        self.cluster_centers = None

    def fit(self, X):
        #
        self.kmeans.fit(X)
        self.labels = self.kmeans.labels_
        self.cluster_centers = self.kmeans.cluster_centers_

    def predict(self, X):
        return self.kmeans.predict(X)

    def predict_proba(self, X):
        labels = self.kmeans.predict(X)
        probabilities = []
        for i in range(X.shape[0]):
            prob = [0] * self.n_clusters
            prob[labels[i]] = 1
            probabilities.append(prob)
        return np.array(probabilities)

    def get_center(self):
        return self.kmeans.cluster_centers_


class KMeansClassifier:
    #This is the base classifier for supervised models, clustering based on sample labels.
    def __init__(self, n_clusters=3, n_runs=10):
        self.n_clusters = n_clusters
        self.n_runs = n_runs

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.clusters = {}

        for c in self.classes:
            X_c = X[y == c]
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_runs)
            kmeans.fit(X_c)
            self.clusters[c] = kmeans

    def predict_proba(self, X):
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, len(self.classes)))

        for i, x in enumerate(X):
            for j, c in enumerate(self.classes):
                clusters = self.clusters[c]
                distances = np.linalg.norm(x - clusters.cluster_centers_, axis=1)
                weights = clusters.labels_.sum(axis=0) / clusters.labels_.size
                probs = 1 / (distances ** 2 + 1e-6)
                proba[i, j] = np.sum(probs * weights)

        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        y_pred = np.argmax(proba, axis=1)
        return self.classes[y_pred]


class SSEA(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, base_classifier=KMeansClassifier(), label_budget=0.1, window_size = 200, L=5, a=5, alpha=2):
        self.base_classifier = base_classifier
        self.window_size = window_size
        self.max_classifier = L
        self.n_unsupervised_models = a
        self.label_budget = label_budget
        self.alpha = alpha

        self.sup_ensemble = []
        self.unsupervised_ensemble = []
        self.X_block = None
        self.Y_block = None
        self.unlabel_data = None
        self.i = -1

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        N, D = X.shape

        if self.i < 0:
            self.X_block = np.zeros((self.window_size, D))
            self.Y_block = np.zeros(self.window_size)
            self.unlabel_data = []
            self.i = 0

        for n in range(N):
            if random.random() < self.label_budget:
                self.X_block[self.i] = X[n]
                self.Y_block[self.i] = y[n]
                self.i = self.i + 1
            else:
                self.unlabel_data.append(X[n])

            if self.i == self.window_size:
                if len(self.sup_ensemble) == 0:
                    for _ in range(self.max_classifier):
                        classifier = cp.deepcopy(self.base_classifier)
                        classifier.fit(self.X_block, self.Y_block)
                        self.sup_ensemble.append(classifier)
                else:
                    classifier = cp.deepcopy(self.base_classifier)
                    classifier.fit(self.X_block, self.Y_block)
                    self.sup_ensemble.append(classifier)

                    index = 0
                    worst = 1
                    # Remove the worst one clustering model from the ensemble.
                    for i in range(len(self.sup_ensemble)):
                        pre = self.sup_ensemble[i].predict(self.X_block)
                        acc = accuracy_score(y_true=self.Y_block, y_pred=pre)
                        if acc<worst:
                            worst = acc
                            index = i
                    self.sup_ensemble.pop(index)
                self.i = 0


            if len(self.unlabel_data) > self.window_size:
                self.unlabel_data.pop(0)

        return self

    def _sup_ensemble_predict_proba(self, X):
        N, D = X.shape
        votes = np.zeros(N)

        for h_i in self.sup_ensemble:
            votes = votes + 1. / len(self.sup_ensemble) * h_i.predict(X)
        return votes


    def _sup_ensemble_predict(self, X):
        votes = self._sup_ensemble_predict_proba(X)
        votes = (votes >= 0.5) * 1.
        return votes

    def predict_proba(self, X):
        N, D = X.shape
        votes = np.zeros(N)
        if len(self.sup_ensemble) <= 0:
            return votes

        if len(self.unlabel_data) < self.window_size:
            for h_i in self.sup_ensemble:
                votes = votes + 1. / len(self.sup_ensemble) * h_i.predict(X)

        unlabel = copy.deepcopy(self.unlabel_data)
        for n in range(N):
            unlabel.append(X[n])
            unlabel.pop(0)
            pre = self._sup_ensemble_predict(np.array(unlabel))
            r = len(np.unique(pre))
            self.unsupervised_ensemble = []
            classifier = KMeansClassifier(n_clusters=2)
            classifier.fit(np.array(unlabel), pre)
            self.unsupervised_ensemble.append(classifier)
            for i in range(self.n_unsupervised_models):
                unsupervised_classifier = KMeansCluster(n_clusters=r)
                unsupervised_classifier.fit(np.array(unlabel))
                self.unsupervised_ensemble.append(unsupervised_classifier)

            votes[n] = self._semi_supervised_learning(np.array(unlabel), self.unsupervised_ensemble, r)

        return votes

    def predict(self, X):
        votes = self.predict_proba(X)
        return (votes >= 0.5) * 1.

    def _semi_supervised_learning(self, unlabel_data, unsupervised_ensemble, r):
        #The key to this paper. It's hard to annotate.
        #In my opinion it is to approximate the output of supervised and unsupervised models and then jointly predict.
        N, D = unlabel_data.shape
        v = r*(1+self.n_unsupervised_models)
        A = unsupervised_ensemble[0].predict_proba(unlabel_data)
        U = unsupervised_ensemble[0].predict_proba(unlabel_data)
        for i in range(len(unsupervised_ensemble) - 1):
            proba = unsupervised_ensemble[i+1].predict_proba(unlabel_data)
            A = np.concatenate((A, proba), axis=1)
            U = U + proba

        A = A / np.sum(A, axis=1, keepdims=True)
        U = U / np.sum(U, axis=1, keepdims=True)

        Y = np.zeros((r, r))
        for i in range(r):
            for j in range(r):
                Y[i, j] = (i == j)*1
        for i in range(len(unsupervised_ensemble) - 1):
            center = unsupervised_ensemble[i+1].get_center()
            pre = unsupervised_ensemble[0].predict(center)
            new = np.zeros((r, r))
            for j in range(r):
                new[j, int(pre[j])] = 1
            Y = np.concatenate((Y, new), axis=0)

        Q = self.calculate_Q(A, U, Y, v, r, N)
        U = self.calculate_U(A, U, Q, v, N)
        befor_loss = self.loss(A, U, Q, Y, v, N)
        difference = 1000000

        while difference > 0.1:
            Q = self.calculate_Q(A, U, Y, v, r, N)
            U = self.calculate_U(A, U, Q, v, N)
            after_loss = self.loss(A, U, Q, Y, v, N)
            difference = befor_loss - after_loss
            befor_loss = after_loss

        return U[-1, -1]

    def calculate_Q(self, A, U, Y, v, r, N):
        Q = np.zeros((v, r))
        for i in range(v):
            q = A[:, i].reshape(N, 1) * U
            q = np.sum(q, axis=0)
            q = q + self.alpha * Y[i]
            q = q / (np.sum(A[:, i]) + self.alpha)
            Q[i] = q
        return Q

    def calculate_U(self, A, U, Q, v, N):
        for i in range(N):
            u = A[i].reshape(v, 1) * Q
            u = np.sum(u, axis=0)
            u = u/np.sum(A[i])
            U[i] = u
        return U

    def loss(self, A, U, Q, Y, v, N):
        loss1 = 0
        for i in range(N):
            for j in range(v):
                loss1 = loss1 + A[i, j] * np.linalg.norm(U[i] - Q[j], ord=2)
        loss2 = 0
        for i in range(v):
            loss2 = loss2 + np.linalg.norm(Q[i] - Y[i], ord=2)
        loss = loss1 + self.alpha * loss2
        return loss












def acc(matrix, data_size):
    sum_value = 0.0
    n, _ = matrix.shape()
    for i in range(n):
        sum_value += matrix.value_at(i, i)
    try:
        return sum_value / data_size
    except ZeroDivisionError:
        return 0



stream = FileStream("SINE.csv")
data, label = stream.next_sample(200)

tree = SSEA()
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
print(acc(matrix,data_size))
plt.show()
