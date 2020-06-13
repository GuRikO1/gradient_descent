import numpy as np


M = 20
N = 30
eps = 0.005

A = np.random.rand(N, M)
AA = np.dot(A.T, A)
hesse = AA
lmd, _ = np.linalg.eig(hesse)

L = np.max(lmd)
step = 1 / L
w_hat = np.ones((M, 1))
b = np.dot(A, w_hat) + np.random.randn(N, 1) / (N * M)


def func(w):
    norm = np.linalg.norm(b - np.dot(A, w))
    return norm * norm


def diffunc(w):
    return 2 * np.dot(AA, w) - 2 * np.dot(A.T, b)


def gradient_discent():
    fw_hat = func(w_hat)
    w = 1 + np.random.rand(M, 1) * 0.1
    fw = func(w)
    print(fw)
    while abs(fw - fw_hat) > eps:
        grad = -diffunc(w)
        w = w + step * grad
        fw = func(w)
        print(abs(fw - fw_hat))


if __name__ == "__main__":
    gradient_discent() 