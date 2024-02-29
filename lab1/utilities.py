import numpy as np
import math


def split_matrix(M):
    n = M.shape[0] // 2
    return M[:n, :n], M[:n, n:], M[n:, :n], M[n:, n:]


def generate_matrix(n):
    return np.random.randint(0.00000001, 1.0, size=(n, n))