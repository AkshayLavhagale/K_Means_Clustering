# k-means clustering aims to partition n observations into k clusters in which each observation belongs to the
# cluster with the nearest mean, serving as a prototype of the cluster.
# If variables are huge, then K-Means most of the times computationally faster than hierarchical clustering.
# difficult to predict k-value = The Elbow Method: plot for WSS[Within-Cluster-Sum of Squared Errors]-vs-k

import numpy as np  # numpy = adding support for large, multi-dimensional arrays and matrices
import sklearn      # sklearn = features various classification, regression and clustering
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# Loading Data set.
digits = load_digits()
data = scale(digits.data)  # this data is our feature.
y = digits.target           # these targets are Labels.

k = 10  # this number reflect amount of centroid we make. can write like k = len(np.unique(y))
samples, features = data.shape

# To score our model we are going to use a function from the sklearn.cluster.KMeans website.
# ReallyDon't know the mathematics behind them.
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))        # euclidean = for elbow method.

# Training the Model
# As this is unsupervised model, we don't give y values to train. It automatically generate y value for every single
# data point. So there is no test or train data here.
clf = KMeans(n_clusters=k, init="random", n_init=10)  # k = centroid, init = amount of time we run algorithm,
bench_k_means(clf, "1", data)
