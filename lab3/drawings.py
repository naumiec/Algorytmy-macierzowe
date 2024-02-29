import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from MatrixTree import MatrixTree



def draw_tree(root: "MatrixTree", title: str="") -> None:
    image = np.ones(root.matrix.shape) * 255
    Q = deque()
    Q.append(root)
    while Q:
        v = Q.pop()
        if v.leaf:
            image[v.row_min:v.row_max, v.col_min:v.col_min + v.rank] = np.zeros((v.row_max - v.row_min, v.rank))
            image[v.row_min:v.row_min + v.rank, v.col_min:v.col_max] = np.zeros((v.rank, v.col_max - v.col_min))
            image[v.row_min, v.col_min:v.col_max] = np.zeros((1, v.col_max - v.col_min))
            image[v.row_max - 1, v.col_min:v.col_max] = np.zeros((1, v.col_max - v.col_min))
            image[v.row_min:v.row_max, v.col_min] = np.zeros(v.row_max - v.row_min)
            image[v.row_min:v.row_max, v.col_max - 1] = np.zeros(v.row_max - v.row_min)
        else:
            for child in v.children:
                Q.append(child)
    plt.imshow(image, cmap="gist_gray", vmin=0, vmax=255)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def draw_sigmas(sigmas: list) -> None:
    new_sigmas = []
    for sigma in sigmas:
        new_sigmas.append(sigma[5:])
    sigmas = new_sigmas
    _, ax = plt.subplots()
    n = len(sigmas[0])
    ax.set_ylabel('Sigma value')
    ax.set_xlabel('Sigma index')
    ax.scatter([i for i in range(n)], sigmas[0], color='green', s=2, label="1% non-zeros")
    ax.scatter([i for i in range(n)], sigmas[1], color='blue', s=2, label="2% non-zeros")
    ax.scatter([i for i in range(n)], sigmas[2], color='purple', s=2, label="5% non-zeros")
    ax.scatter([i for i in range(n)], sigmas[3], color='red', s=2, label="10% non-zeros")
    ax.scatter([i for i in range(n)], sigmas[4], color='yellow', s=2, label="20% non-zeros")
    ax.legend()
    ax.autoscale()
    plt.title("Sigma values for different non-zeros ratios")
    plt.grid()
    plt.autoscale()
    plt.show()
