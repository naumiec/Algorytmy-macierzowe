import numpy as np
import time
import matplotlib.pyplot as plt


def generate_nonzero_matrix(size):
    A = np.random.rand(size, size)
    while np.linalg.matrix_rank(A) < size:
        A = np.random.rand(size, size)
    return A


def inverse_matrix_recursive(A):
    n = A.shape[0]

    if n == 1:
        if  A[0, 0] != 0:
            return np.array([[1 / A[0, 0]]])
        else:
            return np.array([[1 / (A[0, 0] + 1e-10)]])

    A11 = A[:n // 2, :n // 2]
    A12 = A[:n // 2, n // 2:]
    A21 = A[n // 2:, :n // 2]
    A22 = A[n // 2:, n // 2:]

    A11_inv = inverse_matrix_recursive(A11)
    S22 = A22 - np.dot(np.dot(A21, A11_inv), A12)
    S22_inv = inverse_matrix_recursive(S22)

    B11 = np.dot(np.dot(A11_inv, (np.eye(n // 2) + np.dot(np.dot(A12, S22_inv), A21))), A11_inv)
    B12 = -np.dot(np.dot(A11_inv, A12), S22_inv)
    B21 = -np.dot(np.dot(S22_inv, A21), A11_inv)
    B22 = S22_inv

    B = np.vstack((np.hstack((B11, B12)), np.hstack((B21, B22))))

    return B


def count_operations(A):
    n = A.shape[0]

    if n == 1:
        return 1

    return 3 * count_operations(A[:n // 2, :n // 2]) + 2 * count_operations(A[n // 2:, n // 2:]) + 5 * (n // 2) ** 2


sizes = [2 ** i for i in range(0, 13)]
execution_times = []
operations_counts = []

for size in sizes:
    A = generate_nonzero_matrix(size)
    start_time = time.time()
    result = inverse_matrix_recursive(A)
    end_time = time.time()

    execution_time = end_time - start_time
    operations_count = count_operations(A)

    execution_times.append(execution_time)
    operations_counts.append(operations_count)



plt.plot(sizes, execution_times, marker='o')
plt.title('Czas wykonania w zależności od rozmiaru macierzy')
plt.xlabel('Rozmiar macierzy')
plt.ylabel('Czas wykonania (s)')
plt.show()

plt.plot(sizes, operations_counts, marker='o')
plt.title('Liczba operacji w zależności od rozmiaru macierzy')
plt.xlabel('Rozmiar macierzy')
plt.ylabel('Liczba operacji')

plt.show()
