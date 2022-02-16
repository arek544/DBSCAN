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

def check_y(y):
    return True if len(np.unique(y)) > 1 else False


def try_or(func, default=None, expected_exc=(Exception,)):
    try:
        return func()
    except expected_exc:
        return default
        
def evaluate(y_pred, y_true, data):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(y_true)
    y_pred = le.fit_transform(y_pred)
    # y_pred = cluster_encoder(y_pred)

    try:
        davies_bouldin_score_value = davies_bouldin_score(data, y_pred)
    except Exception:
        davies_bouldin_score_value = ""

    try:
        silhouette_score = metrics.silhouette_score(data, y_pred, metric = 'euclidean')
    except Exception:
        silhouette_score = ""
    
    try:
        silhouette_score_cosine = metrics.silhouette_score(data, y_pred, metric = 'cosine')
    except Exception:
        silhouette_score_cosine = ""

    if check_y(y_pred):
        return {
            "purity": purity(y_true, y_pred),
            "rand_score": rand_score(y_true, y_pred),
            # "davies_bouldin_score": davies_bouldin_score_value,
            "silhouette_score_euclidean": silhouette_score,
            "silhouette_score_cosine": silhouette_score_cosine ,
            "TP": tp(y_true, y_pred),
            "TN": tn(y_true, y_pred)
        }
    else:
        return {
            "purity": "",
            "rand_score": "",
            "davies_bouldin_score": "",
            "silhouette_score_euclidean": "",
            "silhouette_score_cosine": "",
            "TP": "",
            "TN": ""
        }