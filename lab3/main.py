from typing import TypeVar
from sklearn.utils.extmath import randomized_svd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from time import perf_counter
import pandas as pd
import sys
from MatrixTree import MatrixTree
from drawings import draw_tree, draw_sigmas



def generate_matrix(n: int, ratio: float) -> np.ndarray:
    size = (n, n)
    array = [0, 1]
    p = [1 - ratio, ratio]
    low = 0.000001
    high = 1
    first = np.random.choice(array, size=size, p=p)
    second = np.random.uniform(low, high, size)
    return np.multiply(first, second)


def main(power: int = 5) -> None:
    l = 0
    ratios = [0.01, 0.02, 0.05, 0.1, 0.2]
    k = 2 ** power
    test_matrices = [generate_matrix(k, r) for r in ratios]
    sigmas = [randomized_svd(m, n_components=k, random_state=0)[1] for m in test_matrices]
    draw_sigmas(sigmas)

    df = pd.DataFrame(columns=['non-zeros', 'b', 'sigma', 'time', 'error'])
    trees = {}

    for index, m in enumerate(test_matrices):
        for b, sigma, sigma_str in \
                [(1, sigmas[index][1], '$\sigma_1$'), (1, sigmas[index][k // 2], '$\sigma_{2^{k-1}}$'),
                 (1, sigmas[index][k - 1], '$\sigma_{2^{k}}$'),
                 (4, sigmas[index][1], '$\sigma_1$'), (4, sigmas[index][k // 2], '$\sigma_{2^{k-1}}$'),
                 (4, sigmas[index][k - 1], '$\sigma_{2^{k}}$')]:
            root = MatrixTree(m, 0, k, 0, k)
            start = perf_counter()
            root.compress(b, sigma)
            end = perf_counter()
            output_matrix = np.zeros(test_matrices[index].shape)
            root.decompress(output_matrix)
            new_row = {'non-zeros': ratios[index], 'b': b, 'sigma': sigma_str, 'time': end - start,
                       'error': np.linalg.norm(output_matrix - test_matrices[index])}
            df.loc[len(df)] = new_row
            trees[(index, b, sigma_str)] = root
    print(df)

    for index, b, sigma in trees:
        draw_tree(trees[(index, b, sigma)],
                  title=f'H-matrix\n{int(ratios[index]*100)}% non-zero | $b={b}$ | $\delta=${sigma}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        main_argument = 10
        main(main_argument)
    else:
        main_argument = int(sys.argv[1])
        main(main_argument)
