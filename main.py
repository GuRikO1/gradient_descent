import numpy as np
import time


M = 200
N = 300
eps = 1e-5

A = np.random.rand(N, M)
AA = np.dot(A.T, A)
hesse = 2 * AA
lmd, _ = np.linalg.eig(hesse)

L = np.max(lmd)
alpha = 1 / L
bete = M / (M + 3)
x_hat = np.ones((M, 1))
b = np.dot(A, x_hat) + np.random.randn(N, 1) / N


def func(x):
    norm = np.linalg.norm(b - np.dot(A, x))
    return norm * norm


def diffunc(x):
    return 2 * np.dot(AA, x) - 2 * np.dot(A.T, b)


def gradient_discent(x):
    fx_hat = func(x_hat)
    fx = func(x)
    while abs(fx - fx_hat) > eps:
        grad = -diffunc(x)
        x = x + alpha * grad
        fx = func(x)


def polyak_momentum(x):
    fx_hat = func(x_hat)
    fx = func(x)
    prev_x = x
    while abs(fx - fx_hat) > eps:
        tmp = x
        x = x - alpha * diffunc(x) + bete * (x - prev_x)
        prev_x = tmp
        fx = func(x)
        print(fx, fx_hat)


def nesterov_acceleration(x):
    fx_hat = func(x_hat)
    fx = func(x)
    y = x
    while abs(fx - fx_hat) > eps:
        prev_x = x
        x = y - alpha * diffunc(y)
        y = x + bete * (x - prev_x)
        fx = func(x)
        print(fx, fx_hat)


if __name__ == "__main__":
    x = 1 + np.random.rand(M, 1) * 0.1

    start = time.time()
    gradient_discent(x)
    t = time.time() - start
    print('gradient_discent {} s'.format(t))

    start = time.time()
    polyak_momentum(x)
    t = time.time() - start
    print('polyak_momentum {} s'.format(t))

    start = time.time()
    nesterov_acceleration(x)
    t = time.time() - start
    print('nesterov_acceleration {} s'.format(t))
