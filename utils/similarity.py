import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def similarity_to_score(sim):
    return round((sim + 1) / 2 * 100, 2)