import numpy as np
from numpy import ndarray
from sklearn.utils.extmath import randomized_svd



class MatrixTree(object):
    def __init__(self, matrix, row_min, row_max, col_min, col_max):
        self.matrix = matrix
        self.row_min = row_min
        self.row_max = row_max
        self.col_min = col_min
        self.col_max = col_max
        self.rank = None
        self.u = None
        self.s = None
        self.v = None
        self.leaf = False
        self.children = []

    def compress(self, r: int, eps: float) -> None:
        M = self.matrix[self.row_min:self.row_max, self.col_min:self.col_max]
        U, Sigma, V = randomized_svd(M, n_components=r + 1, random_state=0)
        if self.row_min + r == self.row_max or Sigma[r] <= eps:
            self.leaf = True
            if not M.any():
                self.rank = 0
            else:
                self.rank = len(Sigma)
                self.u = U
                self.s = Sigma
                self.v = V
        else:
            self.children = []
            new_row_max = (self.row_min + self.row_max) // 2
            new_col_max = (self.col_min + self.col_max) // 2
            self.children.append(MatrixTree(self.matrix, self.row_min, new_row_max, self.col_min, new_col_max))
            self.children.append(MatrixTree(self.matrix, self.row_min, new_row_max, new_col_max, self.col_max))
            self.children.append(MatrixTree(self.matrix, new_row_max, self.row_max, self.col_min, new_col_max))
            self.children.append(MatrixTree(self.matrix, new_row_max, self.row_max, new_col_max, self.col_max))
            for child in self.children:
                child.compress(r, eps)

    def decompress(self, matrix: np.ndarray) -> None:
        if self.leaf:
            if self.rank:
                sigma = np.zeros((self.rank, self.rank))
                np.fill_diagonal(sigma, self.s)
                M = self.u @ sigma @ self.v
                matrix[self.row_min:self.row_max, self.col_min:self.col_max] = M
            else:
                M = self.matrix[self.row_min:self.row_max, self.col_min:self.col_max]
                matrix[self.row_min:self.row_max, self.col_min:self.col_max] = M
        else:
            for child in self.children:
                child.decompress(matrix)

    def get_rank(self) -> int:
        if self.leaf:
            return self.rank
        else:
            return sum([child.get_rank() for child in self.children])

