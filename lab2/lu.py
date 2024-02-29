import numpy as np
import time
import matplotlib.pyplot as plt


def LU_recursive(A):
    n = A.shape[0]
    if n == 1:
        return np.array([[1]]), A

    A11 = A[:n // 2, :n // 2]
    A12 = A[:n // 2, n // 2:]
    A21 = A[n // 2:, :n // 2]
    A22 = A[n // 2:, n // 2:]

    L11, U11 = LU_recursive(A11)
    U11_inv = np.linalg.inv(U11)
    L21 = np.dot(A21, U11_inv)
    L11_inv = np.linalg.inv(L11)
    U12 = np.dot(L11_inv, A12)
    S = A22 - np.dot(np.dot(A21, U11_inv), L11_inv.dot(A12))

    LS, US = LU_recursive(S)

    L = np.block([[L11, np.zeros_like(L11)],
                  [L21, LS]])
    U = np.block([[U11, U12],
                  [np.zeros_like(U12), US]])

    return L, U


def count_operations_LU(A):
    n = A.shape[0]
    if n == 1:
        return 1, 0

    A11 = A[:n // 2, :n // 2]
    A12 = A[:n // 2, n // 2:]
    A21 = A[n // 2:, :n // 2]
    A22 = A[n // 2:, n // 2:]

    ops_L11, ops_U11 = count_operations_LU(A11)
    ops_LS, ops_US = count_operations_LU(A22 - np.dot(np.dot(A21, np.linalg.inv(A11)), np.linalg.inv(A11).dot(A12)))

    total_ops = 2 * ops_L11 + ops_U11 + 2 * ops_LS + ops_US + 2 * (n ** 2 // 4)

    return total_ops, total_ops


# Testowanie dla różnych rozmiarów macierzy i zbieranie danych
sizes = [2 ** i for i in range(0, 13)]
execution_times = []
operations_counts = []

for size in sizes:
    A = np.random.rand(size, size)
    start_time = time.time()
    L, U = LU_recursive(A)
    end_time = time.time()

    execution_time = end_time - start_time
    operations_count_L, operations_count_U = count_operations_LU(A)
    operations_count = operations_count_L + operations_count_U

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
