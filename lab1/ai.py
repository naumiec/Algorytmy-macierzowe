import numpy as np
import math


def match_shape_matrix(M, M_rows, M_cols):
    m, n = M.shape
    height = m // M_rows
    width = n // M_cols

    matrix = np.zeros((m, n))

    for i in range(M_rows):
        for j in range(M_cols):
            matrix[i][j] = M[height * i: height * (i + 1), width * j: width * (j + 1)]

    return matrix


def ai(A, B):
    m, n = A.shape
    n, k = B.shape

    if not (m % 4 == 0 and n % 5 == 0 and k % 5 == 0): return A @ B

    height = m // 4
    width = k // 5

    A_prim = match_shape_matrix(A, 4, 5)
    B_prim = match_shape_matrix(B, 5, 5)

    h = [np.zeros() for _ in range(76)]

    h[0] = ai(A_prim[2][1], -B_prim[1][0] - B_prim[1][4] - B_prim[2][0])
    h[1] = ai(A_prim[1][1] + A_prim[1][4] - A_prim[2][4], -B_prim[1][4] - B_prim[4][0])
    h[2] = ai(-A_prim[2][0] - A_prim[3][0] + A_prim[3][1], -B_prim[0][0] + B_prim[1][4])
    h[3] = ai(A_prim[0][1] + A_prim[0][3] + A_prim[2][3], -B_prim[1][4] - B_prim[3][0])
    h[4] = ai(A_prim[0][4] + A_prim[1][1] + A_prim[1][4], -B_prim[1][3] + B_prim[4][0])
    h[5] = ai(-A_prim[1][1] - A_prim[1][4] - A_prim[3][4], B_prim[1][2] + B_prim[4][0])
    h[6] = ai(-A_prim[0][0] + A_prim[3][0] - A_prim[3][1], B_prim[0][0] + B_prim[1][3])
    h[7] = ai(A_prim[2][1] - A_prim[2][2] - A_prim[3][2], -B_prim[1][2] + B_prim[2][0])
    h[8] = ai(-A_prim[0][1] - A_prim[0][3] + A_prim[3][3], B_prim[1][2] + B_prim[3][0])
    h[9] = ai(A_prim[1][1] + A_prim[1][4], B_prim[4][0])
    h[10] = ai(-A_prim[1][0] - A_prim[3][0] + A_prim[3][1], -B_prim[0][0] + B_prim[1][1])
    h[11] = ai(A_prim[3][0] - A_prim[3][1], B_prim[0][0])
    h[12] = ai(A_prim[0][1] + A_prim[0][3] + A_prim[1][3], B_prim[1][1] + B_prim[3][0])
    h[13] = ai(A_prim[0][2] - A_prim[2][1] + A_prim[2][2], B_prim[1][3] + B_prim[2][0])
    h[14] = ai(-A_prim[0][1] - A_prim[0][3], B_prim[3][0])
    h[15] = ai(-A_prim[2][1] + A_prim[2][2], B_prim[2][0])
    h[16] = ai(A_prim[0][1] + A_prim[0][3] - A_prim[1][0] + A_prim[1][1] - A_prim[1][2] + A_prim[1][3] - A_prim[2][1] + A_prim[2][2] - A_prim[3][0] + A_prim[3][1],
               B_prim[1][1])
    h[17] = ai(A_prim[1][0], B_prim[0][0] + B_prim[0][1] + B_prim[4][1])
    h[18] = ai(-A_prim[1][2], B_prim[2][0] + B_prim[2][1] + B_prim[4][1])
    h[19] = ai(-A_prim[0][4] + A_prim[1][0] + A_prim[1][2] - A_prim[1][4], -B_prim[0][0] - B_prim[0][1] + B_prim[0][3] - B_prim[4][1])
    h[20] = ai(A_prim[1][0] + A_prim[1][2] - A_prim[1][4], B_prim[4][1])
    h[21] = ai(A_prim[0][2] - A_prim[0][3] - A_prim[1][3], B_prim[0][0] + B_prim[0][1] - B_prim[0][3] - B_prim[2][0] - B_prim[2][1] + B_prim[2][3] + B_prim[3][3])
    h[22] = ai(A_prim[0][2], -B_prim[2][0] + B_prim[2][3] + B_prim[3][3])
    h[23] = ai(A_prim[0][4], -B_prim[3][3] - B_prim[4][0] + B_prim[4][3])
    h[24] = ai(-A_prim[0][0], B_prim[0][0] - B_prim[0][3])
    h[25] = ai(-A_prim[0][2] + A_prim[0][3] + A_prim[0][4], B_prim[3][3])
    h[26] = ai(A_prim[0][2] - A_prim[2][0] + A_prim[2][2], B_prim[0][0] - B_prim[0][3] + B_prim[0][4] + B_prim[2][4])
    h[27] = ai(-A_prim[2][3], -B_prim[2][4] - B_prim[3][0] - B_prim[3][4])
    h[28] = ai(A_prim[2][0], B_prim[0][0] + B_prim[0][4] + B_prim[2][4])
    h[29] = ai(A_prim[2][0] - A_prim[2][2] + A_prim[2][3], B_prim[2][4])
    h[30] = ai(-A_prim[0][3] - A_prim[0][4] - A_prim[2][3], -B_prim[3][3] - B_prim[4][0] + B_prim[4][3] - B_prim[4][4])
    h[31] = ai(A_prim[1][0] + A_prim[3][0] + A_prim[3][3], B_prim[0][2] - B_prim[3][0] - B_prim[3][1] - B_prim[3][2])
    h[32] = ai(A_prim[3][2], -B_prim[2][0] - B_prim[2][2])
    h[33] = ai(A_prim[3][3], -B_prim[0][2] + B_prim[3][0] + B_prim[3][2])
    h[34] = ai(-A_prim[3][4], B_prim[0][2] + B_prim[4][0] + B_prim[4][2])
    h[35] = ai(A_prim[1][2] - A_prim[1][4] - A_prim[3][4], B_prim[2][0] + B_prim[2][1] + B_prim[2][2] + B_prim[4][1])
    h[36] = ai(-A_prim[3][0] - A_prim[3][3] + A_prim[3][4], B_prim[0][2])
    h[37] = ai(-A_prim[1][2] - A_prim[2][0] + A_prim[2][2] - A_prim[2][3], B_prim[2][4] + B_prim[3][0] + B_prim[3][1] + B_prim[3][4])
    h[38] = ai(-A_prim[2][0] - A_prim[3][0] - A_prim[3][3] + A_prim[3][4], B_prim[0][2] + B_prim[4][0] + B_prim[4][2] + B_prim[4][4])
    h[39] = ai(-A_prim[0][2] + A_prim[0][3] + A_prim[0][4] - A_prim[3][3], -B_prim[2][0] - B_prim[2][2] + B_prim[2][3] + B_prim[3][3])
    h[40] = ai(-A_prim[0][0] + A_prim[3][0] - A_prim[3][4], B_prim[0][2] + B_prim[2][0] + B_prim[2][2] - B_prim[2][3] + B_prim[4][0] + B_prim[4][2] - B_prim[4][3])
    h[41] = ai(-A_prim[1][0] + A_prim[1][4] - A_prim[2][4], -B_prim[0][0] - B_prim[0][1] - B_prim[0][4] + B_prim[3][0] + B_prim[3][1] + B_prim[3][4] - B_prim[4][1])
    h[42] = ai(A_prim[1][3], B_prim[3][0] + B_prim[3][1])
    h[43] = ai(A_prim[1][2] + A_prim[2][1] - A_prim[2][2], B_prim[1][1] - B_prim[2][0])
    h[44] = ai(-A_prim[2][2] + A_prim[2][3] - A_prim[3][2], B_prim[2][4] + B_prim[3][0] + B_prim[3][2] + B_prim[3][4] + B_prim[4][0] + B_prim[4][2] + B_prim[4][4])
    h[45] = ai(-A_prim[2][4], -B_prim[4][0] - B_prim[4][4])
    h[46] = ai(A_prim[1][0] - A_prim[1][4] - A_prim[2][0] + A_prim[2][4], B_prim[0][0] + B_prim[0][1] + B_prim[0][4] - B_prim[3][0] - B_prim[3][1] - B_prim[3][4])
    h[47] = ai(-A_prim[1][2] + A_prim[2][2], B_prim[1][1] + B_prim[2][1] + B_prim[2][4] + B_prim[3][0] + B_prim[3][1] + B_prim[3][4])
    h[48] = ai(-A_prim[0][0] - A_prim[0][2] + A_prim[0][3] + A_prim[0][4] - A_prim[1][0] - A_prim[1][2] + A_prim[1][3] + A_prim[1][4],
               -B_prim[0][0] - B_prim[0][1] + B_prim[0][3])
    h[49] = ai(-A_prim[0][3] - A_prim[1][3], B_prim[1][1] - B_prim[2][0] - B_prim[2][1] + B_prim[2][3] - B_prim[3][1] + B_prim[3][3])
    h[50] = ai(A_prim[1][1], B_prim[1][0] + B_prim[1][1] - B_prim[4][0])
    h[51] = ai(A_prim[3][1], B_prim[0][0] + B_prim[1][0] + B_prim[1][2])
    h[52] = ai(-A_prim[0][1], -B_prim[1][0] + B_prim[1][3] + B_prim[3][0])
    h[53] = ai(A_prim[0][1] + A_prim[0][3] - A_prim[1][1] - A_prim[1][4] - A_prim[2][1] + A_prim[2][2] - A_prim[3][1] + A_prim[3][2] - A_prim[3][3] - A_prim[3][4],
               B_prim[1][2])
    h[54] = ai(A_prim[0][3] - A_prim[3][3], -B_prim[1][2] + B_prim[2][0] + B_prim[2][2] - B_prim[2][3] + B_prim[3][2] - B_prim[3][3])
    h[55] = ai(A_prim[0][0] - A_prim[0][4] - A_prim[3][0] + A_prim[3][4], B_prim[2][0] + B_prim[2][2] - B_prim[2][3] + B_prim[4][0] + B_prim[4][2] - B_prim[4][3])
    h[56] = ai(-A_prim[2][0] - A_prim[3][0], -B_prim[0][2] - B_prim[0][4] - B_prim[1][4] - B_prim[4][0] - B_prim[4][2] - B_prim[4][4])
    h[57] = ai(-A_prim[0][3] - A_prim[0][4] - A_prim[2][3] - A_prim[2][4], -B_prim[4][0] + B_prim[4][3] - B_prim[4][4])
    h[58] = ai(-A_prim[2][2] + A_prim[2][3] - A_prim[3][2] + A_prim[3][3], B_prim[3][0] + B_prim[3][2] + B_prim[3][4] + B_prim[4][0] + B_prim[4][2] + B_prim[4][4])
    h[59] = ai(A_prim[1][4] + A_prim[3][4], B_prim[1][2] - B_prim[2][0] - B_prim[2][1] - B_prim[2][2] - B_prim[4][1] - B_prim[4][2])
    h[60] = ai(A_prim[0][3] + A_prim[2][3],
               B_prim[0][0] - B_prim[0][3] + B_prim[0][4] - B_prim[1][4] - B_prim[3][3] + B_prim[3][4] - B_prim[4][0] + B_prim[4][3] - B_prim[4][4])
    h[61] = ai(A_prim[1][0] + A_prim[3][0], B_prim[0][1] + B_prim[0][2] + B_prim[1][1] - B_prim[3][0] - B_prim[3][1] - B_prim[3][2])
    h[62] = ai(-A_prim[2][2] - A_prim[3][2], -B_prim[1][2] - B_prim[2][2] - B_prim[2][4] - B_prim[3][0] - B_prim[3][2] - B_prim[3][4])
    h[63] = ai(A_prim[0][0] - A_prim[0][2] - A_prim[0][3] + A_prim[2][0] - A_prim[2][2] - A_prim[2][3], B_prim[0][0] - B_prim[0][3] + B_prim[0][4])
    h[64] = ai(-A_prim[0][0] + A_prim[3][0], -B_prim[0][2] + B_prim[0][3] + B_prim[1][3] - B_prim[4][0] - B_prim[4][2] + B_prim[4][3])
    h[65] = ai(A_prim[0][0] - A_prim[0][1] + A_prim[0][2] - A_prim[0][4] - A_prim[1][1] - A_prim[1][4] - A_prim[2][1] + A_prim[2][2] - A_prim[3][0] + A_prim[3][1],
               B_prim[1][3])
    h[66] = ai(A_prim[1][4] - A_prim[2][4],
               B_prim[0][0] + B_prim[0][1] + B_prim[0][4] - B_prim[1][4] - B_prim[3][0] - B_prim[3][1] - B_prim[3][4] + B_prim[4][1] + B_prim[4][4])
    h[67] = ai(A_prim[0][0] + A_prim[0][2] - A_prim[0][3] - A_prim[0][4] - A_prim[3][0] - A_prim[3][2] + A_prim[3][3] + A_prim[3][4],
               -B_prim[2][0] - B_prim[2][2] + B_prim[2][3])
    h[68] = ai(-A_prim[0][2] + A_prim[0][3] - A_prim[1][2] + A_prim[1][3], -B_prim[1][3] - B_prim[2][0] - B_prim[2][1] + B_prim[2][3] - B_prim[4][1] + B_prim[4][3])
    h[69] = ai(A_prim[1][2] - A_prim[1][4] + A_prim[3][2] - A_prim[3][4], -B_prim[2][0] - B_prim[2][1] - B_prim[2][2])
    h[70] = ai(-A_prim[2][0] + A_prim[2][2] - A_prim[2][3] + A_prim[2][4] - A_prim[3][0] + A_prim[3][2] - A_prim[3][3] + A_prim[3][4],
               -B_prim[4][0] - B_prim[4][2] - B_prim[4][4])
    h[71] = ai(-A_prim[1][0] - A_prim[1][3] - A_prim[3][0] - A_prim[3][3], B_prim[3][0] + B_prim[3][1] + B_prim[3][2])
    h[72] = ai(A_prim[0][2] - A_prim[0][3] - A_prim[0][4] + A_prim[1][2] - A_prim[1][3] - A_prim[1][4],
               B_prim[0][0] + B_prim[0][1] - B_prim[0][3] + B_prim[1][3] + B_prim[4][1] - B_prim[4][3])
    h[73] = ai(A_prim[1][0] - A_prim[1][2] + A_prim[1][3] - A_prim[2][0] + A_prim[2][2] - A_prim[2][3], B_prim[3][0] + B_prim[3][1] + B_prim[3][4])
    h[74] = ai(-A_prim[0][1] - A_prim[0][3] + A_prim[1][1] + A_prim[1][4] + A_prim[2][0] - A_prim[2][1] - A_prim[2][3] - A_prim[2][4] + A_prim[3][0] - A_prim[3][1],
               B_prim[1][4])
    h[75] = ai(A_prim[0][2] + A_prim[2][2], -B_prim[0][0] + B_prim[0][3] - B_prim[0][4] + B_prim[1][3] + B_prim[2][3] - B_prim[2][4])

    C = [[np.zeros() for _ in range(5)] for _ in range(4)]

    count_plus, count_times = 168 * height * width + 192 * width * width, 0

    for i in range(76):
        count_plus += h[i][1]
        count_times += h[i][2]

        h[i] = h[i][0]

    C[0][0] = -h[9] + h[11] + h[13] - h[14] - h[15] + h[52] + h[4] - h[65] - h[6]
    C[1][0] = h[9] + h[10] - h[11] + h[12] + h[14] + h[15] - h[16] - h[43] + h[50]
    C[2][0] = h[9] - h[11] + h[14] + h[15] - h[0] + h[1] + h[2] - h[3] + h[74]
    C[3][0] = -h[9] + h[11] - h[14] - h[15] + h[51] + h[53] - h[5] - h[7] + h[8]
    C[0][1] = h[12] + h[14] + h[19] + h[20] - h[21] + h[22] + h[24] - h[42] + h[48] + h[49]
    C[1][1] = -h[10] + h[11] - h[12] - h[14] - h[15] + h[16] + h[17] - h[18] - h[20] + h[42] + h[43]
    C[2][1] = -h[15] - h[18] - h[20] - h[27] - h[28] - h[37] + h[41] + h[43] - h[46] + h[47]
    C[3][1] = h[10] - h[11] - h[17] + h[20] - h[31] + h[32] - h[33] - h[35] + h[61] - h[69]
    C[0][2] = h[14] + h[22] + h[23] + h[33] - h[36] + h[39] - h[40] + h[54] - h[55] - h[8]
    C[1][2] = -h[9] + h[18] + h[31] + h[34] + h[35] + h[36] - h[42] - h[59] - h[5] - h[71]
    C[2][2] = -h[15] - h[27] + h[32] + h[36] - h[38] + h[44] - h[45] + h[62] - h[70] - h[7]
    C[3][2] = h[9] + h[14] + h[15] - h[32] + h[33] - h[34] - h[36] - h[53] + h[5] + h[7] - h[8]
    C[0][3] = -h[9] + h[11] + h[13] - h[15] + h[22] + h[23] + h[24] + h[25] + h[4] - h[65] - h[6]
    C[1][3] = h[9] + h[17] - h[18] + h[19] - h[21] - h[23] - h[25] - h[4] - h[68] + h[72]
    C[2][3] = -h[13] + h[15] - h[22] - h[25] + h[26] + h[28] + h[30] + h[45] - h[57] + h[75]
    C[3][3] = h[11] + h[24] + h[25] - h[32] - h[34] - h[39] + h[40] + h[64] - h[67] - h[6]
    C[0][4] = h[14] + h[23] + h[24] + h[26] - h[27] + h[29] + h[30] - h[3] + h[60] + h[63]
    C[1][4] = -h[9] - h[17] - h[1] - h[29] - h[37] + h[41] - h[42] + h[45] + h[66] + h[73]
    C[2][4] = -h[9] + h[11] - h[14] + h[27] + h[28] - h[1] - h[29] - h[2] + h[45] + h[3] - h[74]
    C[3][4] = -h[11] - h[28] + h[29] - h[33] + h[34] + h[38] + h[2] - h[44] + h[56] + h[58]

    count_plus += 180 * height * width

    C = np.zeros((m, k))
    for i in range(4):
        for j in range(5):
            C[i * height: (i + 1) * height, j * width: (j + 1) * width] = C[i][j]

    return C, count_plus, count_times


def test_ai():
    A = np.random.randint(0, 10, (8, 8))
    B = np.random.randint(0, 10, (8, 8))
    C = ai(A, B)
    print(C)
    print(A @ B)


test_ai()
