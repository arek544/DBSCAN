import time
import warnings
import math
from scipy.spatial import distance

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn import cluster

np.random.seed(0)

from sklearn.datasets import make_blobs
data, y = make_blobs(n_samples=200, centers = [(-5, -5), (0, 0), (5, 5)], cluster_std=1, n_features=2, random_state=5)
data = StandardScaler().fit_transform(data)

fig, ax = plt.subplots()
ax.scatter(data[:,0], data[:,1])
for i, txt in enumerate(range(len(data))):
    ax.annotate(i, (data[i,0], data[i,1]), size=15)
plt.savefig("./img/test.png")


# ---------------------------------------------------------

n = data.shape[0]
cluster = np.array([0] * n)
# state = np.array([NOT_VISITED] * n)
# cluster_id = 1

all_point_indices = list(range(len(data)))

def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)

def get_knn(current_index, neighbor_indices, k):
    neighbor_indices.remove(current_index)
    neighbor_distances = []
    for neighbor_index in neighbor_indices:
        distance = euclidean_distance(data[neighbor_index], data[current_index])
        neighbor_distances.append(distance) 
    sort_indices = np.argsort(neighbor_distances)
    neighbor_indices = np.array(neighbor_indices)
    return neighbor_indices[sort_indices][:k].tolist()

def get_rnn(point_knn, current_index):
    rnn = []
    for neighbor in point_knn[current_index]:
        if current_index in point_knn[neighbor]:
            rnn.append(neighbor)
    return rnn
    
point_rnn = {}
point_knn = {}

S_non_core = []
S_core = []

k = 40

for current_index in all_point_indices:
    knn = get_knn(current_index, all_point_indices[:], k)
    point_knn[current_index] = knn
    
for current_index in point_knn.keys():
    rnn = get_rnn(point_knn, current_index)
    point_rnn[current_index] = rnn

# ----------------------------------------------------------
NOT_VISITED = 0
VISTED = 1
CLUSTERED = 2


n = data.shape[0]
cluster = np.array([0] * n)
state = np.array([NOT_VISITED] * n) # if core point
cluster_id = 1

def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)

def search(current_index, k):
    if len(point_rnn[current_index]) < k:
            state[current_index] = VISTED
    else:
        state[current_index] = CLUSTERED
        cluster[current_index] = cluster_id
        for neighbor_index in point_rnn[current_index]:
            if state[neighbor_index] == NOT_VISITED:
                state[neighbor_index] = CLUSTERED
                cluster[neighbor_index] = cluster_id
                search(neighbor_index, k)

while NOT_VISITED in state:
    not_visited_ids = np.where(state==NOT_VISITED)[0]
    search(not_visited_ids[0], k)
    cluster_id += 1

from collections import Counter

while VISTED in state:
    visited_id = np.where(state==VISTED)[0][0]
    knn = get_knn(visited_id, all_point_indices[:], k)
    cluster[visited_id] = Counter(cluster[knn]).most_common(1)[0][0] 
    state[visited_id] = CLUSTERED

# ---------------------------------------------------------

fig, ax = plt.subplots()
ax.scatter(data[:,0], data[:,1], c=cluster)
# for i, txt in enumerate(range(len(data))):
#     # ax.annotate(i, (data[i,0], data[i,1]), size=15)
#     ax.annotate(cluster[i], (data[i,0], data[i,1]), size=15)
plt.savefig("./img/test2.png")

print(cluster)