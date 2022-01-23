import numpy as np

def euclidean_distance(a, b):
    similarity_measure = 0
    for i in range(len(a)):
         similarity_measure += pow((a[i] - b[i]), 2)
    return np.sqrt(similarity_measure)

def cosine_dissimilarity(a, b):
    dot = 0
    norm_a = 0
    norm_b = 0
    for i in range(len(a)):
        dot += a[i] * b[i]
        norm_a += a[i]**2
        norm_b += b[i]**2
    return 1 - dot / (norm_a * norm_b)**0.5

def cosine_similarity(a, b):
    dot = 0
    norm_a = 0
    norm_b = 0
    for i in range(len(a)):
        dot += a[i] * b[i]
        norm_a += a[i]**2
        norm_b += b[i]**2
    return dot / (norm_a * norm_b)**0.5