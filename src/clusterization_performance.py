from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
from sklearn import preprocessing
import numpy as np
import math


def cluster_encoder(cluster):
    le = preprocessing.LabelEncoder()
    le.fit(cluster)
    return le.transform(cluster)

# n_clusters: the number of discovered clusters,
# y: real clusters,
# c: discovered clusters

# Purity
def purity(C, G):
    common_items_inclusters = metrics.cluster.contingency_matrix(C, G)
    purity_score = np.sum(np.amax(common_items_inclusters, axis=0)) / np.sum(common_items_inclusters)
    return purity_score

def tp(y_true, y_pred):
    return sum(y_true == y_pred)
    
def tn(y_true, y_pred):
    return sum(y_true != y_pred)

def rand_score(y_true, y_pred):
    n = len(y_true)
    return (tp(y_true, y_pred) + tn(y_true, y_pred))/math.comb(n,2)

    
def evaluate(y_pred, y_true, data):
    # y_pred = cluster_encoder(y_pred)
    return {
        "purity": purity(y_true, y_pred),
        "rand_score": rand_score(y_true, y_pred),
        "davies_bouldin_score": davies_bouldin_score(data, y_pred),
        "silhouette_score_euclidean": metrics.silhouette_score(data, y_pred, metric = 'euclidean'),
        "silhouette_score_cosine": metrics.silhouette_score(data, y_pred, metric = 'cosine'),
        "TP": tp(y_true, y_pred),
        "TN": tn(y_true, y_pred)
    }