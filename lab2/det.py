import numpy as np
import time
import matplotlib.pyplot as plt


def LU_recursive(A):
    n = A.shape[0]
    if n == 1:
        return A, A, 0

    count_ops = 0

    A11 = A[:n // 2, :n // 2]
    A12 = A[:n // 2, n // 2:]
    A21 = A[n // 2:, :n // 2]
    A22 = A[n // 2:, n // 2:]

    L11, U11, ops1 = LU_recursive(A11)
    count_ops += ops1
    count_ops += 2 * (n // 2) ** 2  # For copying blocks

    U11_inv = np.linalg.inv(U11)
    count_ops += 2 * (n // 2) ** 3  # For inverting U11

    L21 = np.dot(A21, U11_inv)
    count_ops += 2 * (n // 2) ** 3  # For matrix multiplication

    L11_inv = np.linalg.inv(L11)
    count_ops += 2 * (n // 2) ** 3  # For inverting L11

    U12 = np.dot(L11_inv, A12)
    count_ops += 2 * (n // 2) ** 3  # For matrix multiplication

    S = A22 - np.dot(np.dot(A21, U11_inv), L11_inv.dot(A12))
    count_ops += 2 * (n // 2) ** 3  # For matrix multiplication and subtraction

    LS, US, ops2 = LU_recursive(S)
    count_ops += ops2

    L = np.block([[L11, np.zeros_like(L11)],
                  [L21, LS]])
    U = np.block([[U11, U12],
                  [np.zeros_like(U12), US]])

    return L, U, count_ops


def determinant_recursive(A):
    L, U, count_ops = LU_recursive(A)
    det_U = np.prod(np.diag(U))
    return det_U, count_ops


# Testowanie dla różnych rozmiarów macierzy i zbieranie danych
sizes = [2 ** i for i in range(0, 13)]
execution_times = []
operations_counts = []

for size in sizes:
    A = np.random.rand(size, size)
    start_time = time.time()
    determinant, ops = determinant_recursive(A)
    end_time = time.time()

    execution_time = end_time - start_time

    execution_times.append(execution_time)
    operations_counts.append(ops)

# Wykresy

plt.plot(sizes, execution_times, marker='o')
plt.title('Czas wykonania obliczania wyznacznika')
plt.xlabel('Rozmiar macierzy')
plt.ylabel('Czas wykonania (s)')
plt.show()

plt.plot(sizes, operations_counts, marker='o')
plt.title('Liczba operacji w obliczaniu wyznacznika')
plt.xlabel('Rozmiar macierzy')
plt.ylabel('Liczba operacji')
plt.show()

plt.show()
