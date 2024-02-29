import numpy as np
import math
from utilities import split_matrix


def strassen(A, B):
    n = A.shape[0]
    if n == 1:
        return A * B

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11, B12 - B22)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    return C


def strassen_with_count(A, B):
    if A.shape[0] == 1:
        return A * B, 1

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    M1, count1 = strassen_with_count(A11 + A22, B11 + B22)
    M2, count2 = strassen_with_count(A21 + A22, B11)
    M3, count3 = strassen_with_count(A11, B12 - B22)
    M4, count4 = strassen_with_count(A22, B21 - B11)
    M5, count5 = strassen_with_count(A11 + A12, B22)
    M6, count6 = strassen_with_count(A21 - A11, B11 + B12)
    M7, count7 = strassen_with_count(A12 - A22, B21 + B22)

    count = count1 + count2 + count3 + count4 + count5 + count6 + count7 + 18 * math.prod(A.shape)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    return C, count


def test_strassen():
    A = np.random.randint(0, 10, (4, 4))
    B = np.random.randint(0, 10, (4, 4))
    result, count = strassen_with_count(A, B)
    expected_result = A @ B
    print(A)
    print(B)
    print(result)
    print(expected_result)
    assert np.allclose(result, expected_result, rtol=1e-05, atol=1e-08, equal_nan=False)


test_strassen()
