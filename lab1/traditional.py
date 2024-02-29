def matrix_multiplication(first, second):
    if len(first[0]) != len(second):
        raise ValueError("Number of columns in the first matrix must match the number of rows in the second matrix")

    result = [[0 for _ in range(len(second[0]))] for _ in range(len(first))]

    for i in range(len(first)):
        for j in range(len(second[0])):
            sum = 0
            for k in range(len(second)):
                sum += first[i][k] * second[k][j]
            result[i][j] = sum

    return result


def test_traditional(A, B):
    '''
    first_matrix = [
        [1, 2, 3],
        [4, 5, 6],
    ]

    second_matrix = [
        [7, 8],
        [9, 10],
        [11, 12],
    ]

    result = matrix_multiplication(first_matrix, second_matrix)
    print(result)
    '''

    result = matrix_multiplication(A, B)
    print(result)


