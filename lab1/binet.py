import numpy as np
import math


def split_matrix(M):
    n = M.shape[0] // 2
    return M[:n, :n], M[:n, n:], M[n:, :n], M[n:, n:]


def binet(A, B):
    n = A.shape[0]
    if n == 1:
        return A * B
    else:
        A11, A12, A21, A22 = split_matrix(A)
        B11, B12, B21, B22 = split_matrix(B)

        C1 = binet(A11, B11) + binet(A12, B21)
        C2 = binet(A11, B12) + binet(A12, B22)
        C3 = binet(A21, B11) + binet(A22, B21)
        C4 = binet(A21, B12) + binet(A22, B22)

        return np.vstack((np.hstack((C1, C2)), np.hstack((C3, C4))))


def binet_with_count(A, B):
    n = A.shape[0]
    if n == 1:
        return A * B, 1
    else:
        A11, A12, A21, A22 = split_matrix(A)
        B11, B12, B21, B22 = split_matrix(B)

        C11, count1 = binet_with_count(A11, B11)
        C12, count2 = binet_with_count(A11, B12)
        C21, count3 = binet_with_count(A21, B11)
        C22, count4 = binet_with_count(A21, B12)
        C1 = C11 + binet_with_count(A12, B21)[0]
        C2 = C12 + binet_with_count(A12, B22)[0]
        C3 = C21 + binet_with_count(A22, B21)[0]
        C4 = C22 + binet_with_count(A22, B22)[0]

        count = count1 + count2 + count3 + count4 + 4 * n * n

        return np.vstack((np.hstack((C1, C2)), np.hstack((C3, C4)))), count


def test_binet():
    A = np.random.randint(0, 10, (4, 4))
    B = np.random.randint(0, 10, (4, 4))
    result, count = binet_with_count(A, B)
    expected_result = A @ B
    print(A)
    print(B)
    print(result)
    print(count)
    print(expected_result)
    assert np.allclose(result, expected_result, rtol=1e-05, atol=1e-08, equal_nan=False)


test_binet()
