import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pandas as pd

from utilities import split_matrix, generate_matrix
from traditional import matrix_multiplication
from strassen import strassen, strassen_with_count
from binet import binet, binet_with_count
from ai import ai


def perform_test(end = 8):
    n = [2 ** i for i in range(0, end)]
    times_strassen = []
    times_binet = []
    times_ai = []

    for i in n:
        A = generate_matrix(i)
        B = generate_matrix(i)
        '''
        C = A @ B

        C_strassen, count = strassen_with_count(A, B)
        C_binet, count = binet_with_count(A, B)
        C_ai, count = ai(A, B)

        print(f"n = {i}")
        print(f"Strassen: {np.allclose(C, C_strassen)}")
        print(f"Binet: {np.allclose(C, C_binet)}")
        print(f"AI: {np.allclose(C, C_ai)}")
        print()
        '''


        start = time.time()
        strassen(A, B)
        end = time.time()
        times_strassen.append(end - start)
        print(f"Strassen: {end - start}")

        start = time.time()
        binet(A, B)
        end = time.time()
        times_binet.append(end - start)
        print(f"Binet: {end - start}")

        start = time.time()
        ai(A, B)
        end = time.time()
        times_ai.append(end - start)
        print(f"AI: {end - start}")

        print(f"[{i}]")

    with open('times.txt', 'w') as file:
        file.write(f"n\tstrassen\tbinet\tai\n")
        for i in range(len(n)):
            file.write(f"{n[i]}\t{times_strassen[i]}\t{times_binet[i]}\t{times_ai[i]}\n")


    df = pd.DataFrame({'n': n, 'strassen': times_strassen, 'binet': times_binet, 'ai': times_ai})
    df.to_csv('times.csv', index=False)
    plt.plot(n, times_strassen, label='Strassen')
    plt.plot(n, times_binet, label='Binet')
    plt.plot(n, times_ai, label='AI')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.legend()
    plt.show()


def plot_from_file(file_name='times.csv'):
    df = pd.read_csv(file_name)
    n = df['n']
    times_strassen = df['strassen']
    times_binet = df['binet']
    times_ai = df['ai']
    plt.plot(n, times_strassen, label='Strassen')
    plt.plot(n, times_binet, label='Binet')
    plt.plot(n, times_ai, label='AI')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.legend()
    plt.show()
    plt.savefig('times.png')


def plot_strassen(end = 8):
    n = [2 ** i for i in range(0, end)]
    times_strassen = []

    for i in n:
        A = generate_matrix(i)
        B = generate_matrix(i)

        start = time.time()
        strassen(A, B)
        end = time.time()
        times_strassen.append(end - start)
        print(f"Strassen: {end - start}")

        print(f"[{i}]")

    with open('times_strassen.txt', 'w') as file:
        file.write(f"n\tstrassen\n")
        for i in range(len(n)):
            file.write(f"{n[i]}\t{times_strassen[i]}\n")

    df = pd.DataFrame({'n': n, 'strassen': times_strassen})
    df.to_csv('times_strassen.csv', index=False)
    plt.plot(n, times_strassen, label='Strassen')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.legend()
    plt.show()
    plt.savefig('times_strassen.png')


def plot_binet(end = 8):
    n = [2 ** i for i in range(0, end)]
    times_binet = []

    for i in n:
        A = generate_matrix(i)
        B = generate_matrix(i)

        start = time.time()
        binet(A, B)
        end = time.time()
        times_binet.append(end - start)
        print(f"Binet: {end - start}")

        print(f"[{i}]")

    with open('times_binet.txt', 'w') as file:
        file.write(f"n\tbinet\n")
        for i in range(len(n)):
            file.write(f"{n[i]}\t{times_binet[i]}\n")

    df = pd.DataFrame({'n': n, 'binet': times_binet})
    df.to_csv('times_binet.csv', index=False)
    plt.plot(n, times_binet, label='Binet')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.legend()
    plt.show()
    plt.savefig('times_binet.png')


def plot_ai(end = 8):
    n = [2 ** i for i in range(0, end)]
    times_ai = []

    for i in n:
        A = generate_matrix(i)
        B = generate_matrix(i)

        start = time.time()
        ai(A, B)
        end = time.time()
        times_ai.append(end - start)
        print(f"AI: {end - start}")

        print(f"[{i}]")

    with open('times_ai.txt', 'w') as file:
        file.write(f"n\tai\n")
        for i in range(len(n)):
            file.write(f"{n[i]}\t{times_ai[i]}\n")

    df = pd.DataFrame({'n': n, 'ai': times_ai})
    df.to_csv('times_ai.csv', index=False)
    plt.plot(n, times_ai, label='AI')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.legend()
    plt.show()
    plt.savefig('times_ai.png')

def plot_traditional(end = 8):
    n = [2 ** i for i in range(0, end)]
    times_traditional = []

    for i in n:
        A = generate_matrix(i)
        B = generate_matrix(i)

        start = time.time()
        matrix_multiplication(A, B)
        end = time.time()
        times_traditional.append(end - start)
        print(f"Traditional: {end - start}")

        print(f"[{i}]")

    with open('times_traditional.txt', 'w') as file:
        file.write(f"n\ttraditional\n")
        for i in range(len(n)):
            file.write(f"{n[i]}\t{times_traditional[i]}\n")

    df = pd.DataFrame({'n': n, 'traditional': times_traditional})
    df.to_csv('times_traditional.csv', index=False)
    plt.plot(n, times_traditional, label='Traditional')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.legend()
    plt.show()
    plt.savefig('times_traditional.png')


def plot_traditional_binet_strassen(end = 8):
    n = [2 ** i for i in range(0, end)]
    times_traditional = []
    times_binet = []
    times_strassen = []

    for i in n:
        A = generate_matrix(i)
        B = generate_matrix(i)

        start = time.time()
        A @ B
        end = time.time()
        times_traditional.append(end - start)
        print(f"Traditional: {end - start}")

        start = time.time()
        binet(A, B)
        end = time.time()
        times_binet.append(end - start)
        print(f"Binet: {end - start}")

        start = time.time()
        strassen(A, B)
        end = time.time()
        times_strassen.append(end - start)
        print(f"Strassen: {end - start}")

        print(f"[{i}]")

    with open('times_traditional_binet_strassen.txt', 'w') as file:
        file.write(f"n\ttraditional\tbinet\tstrassen\n")
        for i in range(len(n)):
            file.write(f"{n[i]}\t{times_traditional[i]}\t{times_binet[i]}\t{times_strassen[i]}\n")

    df = pd.DataFrame({'n': n, 'traditional': times_traditional, 'binet': times_binet, 'strassen': times_strassen})
    df.to_csv('times_traditional_binet_strassen.csv', index=False)
    plt.plot(n, times_traditional, label='Traditional')
    plt.plot(n, times_binet, label='Binet')
    plt.plot(n, times_strassen, label='Strassen')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.legend()
    plt.show()
    plt.savefig('times_traditional_binet_strassen.png')


def plot_binet_strassen(end = 8):
    n = [2 ** i for i in range(0, end)]
    times_binet = []
    times_strassen = []

    for i in n:
        A = generate_matrix(i)
        B = generate_matrix(i)

        start = time.time()
        binet(A, B)
        end = time.time()
        times_binet.append(end - start)
        print(f"Binet: {end - start}")

        start = time.time()
        strassen(A, B)
        end = time.time()
        times_strassen.append(end - start)
        print(f"Strassen: {end - start}")

        print(f"[{i}]")

    with open('times_binet_strassen.txt', 'w') as file:
        file.write(f"n\tbinet\tstrassen\n")
        for i in range(len(n)):
            file.write(f"{n[i]}\t{times_binet[i]}\t{times_strassen[i]}\n")

    df = pd.DataFrame({'n': n, 'binet': times_binet, 'strassen': times_strassen})
    df.to_csv('times_binet_strassen.csv', index=False)
    plt.plot(n, times_binet, label='Binet')
    plt.plot(n, times_strassen, label='Strassen')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.legend()
    plt.show()
    plt.savefig('times_binet_strassen.png')


if __name__ == "__main__":
    #perform_test()
    #plot_from_file()
    #plot_strassen(9)
    #plot_binet(9)
    #plot_ai(9)
    #plot_traditional(9)
    #plot_traditional_binet_strassen(9)
    plot_binet_strassen(9)